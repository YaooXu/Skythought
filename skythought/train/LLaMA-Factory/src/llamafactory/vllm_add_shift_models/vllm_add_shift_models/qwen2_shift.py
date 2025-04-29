# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from ast import Pass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP, HasInnerState
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

import torch.nn.functional as F
import os

from transformers import PretrainedConfig

logger = init_logger(__name__)

class LastHiddenStateCacheManager:

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        max_batch_size: int,
        config: PretrainedConfig,
    ):
        self.device = device
        self.hidden_size = config.hidden_size
        last_kv_size = max_batch_size * 2
        self.max_batch_size = max_batch_size
        self.num_layers = config.num_hidden_layers

        self.last_hidden_state_caches = []
        
        for _ in range(self.num_layers):
            self.last_hidden_state_caches.append(
                torch.zeros((last_kv_size, self.hidden_size), dtype=dtype, device=device)
            )

        self.cache_indices_mapping: Dict[str, int] = {}
        self.free_cache_indices = list(range(max_batch_size))

    def _release_finished_requests(self, finished_req_ids: List[str]):
        for req_id in finished_req_ids:
            if req_id in self.cache_indices_mapping:
                index = self.cache_indices_mapping.pop(req_id)
                self.free_cache_indices.append(index)

    def _get_cache_indices(self, request_ids_to_seq_ids, finished_requests_ids):
        self._release_finished_requests(finished_requests_ids)
        indices = [0] * len(request_ids_to_seq_ids)
        for i, (req_id, _) in enumerate(request_ids_to_seq_ids.items()):
            if req_id in self.cache_indices_mapping:
                indices[i] = self.cache_indices_mapping[req_id]
            elif req_id in finished_requests_ids:
                indices[i] = 0  # warmup
            else:
                assert len(self.free_cache_indices) > 0
                index = self.free_cache_indices.pop()
                self.cache_indices_mapping[req_id] = index
                indices[i] = index
        return indices

    def get_last_hidden_states(self, request_ids_to_seq_ids, finished_requests_ids):
        indices = self._get_cache_indices(request_ids_to_seq_ids, finished_requests_ids)
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        return self.last_hidden_state_caches, indices_tensor  # shape: [num_layers][B, H]

    def update_last_hidden_states(self, hidden_states_by_layer, indices_tensor):
        """
        hidden_states_by_layer: list of tensors, each is [B, H]
        indices_tensor: [B], indicating the slot to write
        """
        for layer_id, layer_tensor in enumerate(self.last_hidden_state_caches):
            layer_tensor.index_copy_(0, indices_tensor, hidden_states_by_layer[layer_id])

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant state_indices into the CUDA graph input buffer.
        This is used before graph replay to populate slot indices.
        """
        assert all(
            key in kwargs
            for key in ["request_ids_to_seq_ids", "finished_requests_ids"]
        ), "CUDA Graph mode requires explicit cache index resolution."

        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
        finished_requests_ids = kwargs["finished_requests_ids"]

        assert "seqlen_agnostic_capture_inputs" in input_buffers
        _, input_state_indices_buffer = input_buffers["seqlen_agnostic_capture_inputs"]

        # 获取当前 batch 的缓存槽位索引
        last_kv_indices = self._get_cache_indices(
            request_ids_to_seq_ids, finished_requests_ids
        )

        # CUDA Graph buffer padding（补齐静态 shape）
        cuda_graph_pad_len = input_state_indices_buffer.shape[0] - len(last_kv_indices)
        last_kv_indices.extend(
            list(range(self.max_batch_size, self.max_batch_size + cuda_graph_pad_len))
        )

        # 写入 capture buffer
        input_state_indices_buffer.copy_(
            torch.as_tensor(last_kv_indices, dtype=torch.int32, device=self.device)
        )


    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Used during CUDA graph capture to provide static buffers:
        - cache tensors: list of [max_batch_size, hidden_size] per layer
        - state indices tensor: [batch_size]
        """
        state_indices_tensor = torch.as_tensor(
            list(range(0, batch_size)), dtype=torch.int32, device=self.device
        )
        return (self.last_hidden_state_caches, state_indices_tensor)


def get_prev_hidden_with_zero_start(
    hidden_states: torch.Tensor,                # [T, D]
    query_start_loc: torch.Tensor,              # [num_prefills + 1], int32
) -> torch.Tensor:
    """
    构造 prev_hidden:
    - 起始 token → 0 向量
    - 其他 token → 前一个 token 的 hidden
    """
    device = hidden_states.device
    num_prefill_tokens = query_start_loc[-1].item()
    D = hidden_states.shape[1]

    # 结果张量
    prev_hidden = torch.zeros((num_prefill_tokens, D), dtype=hidden_states.dtype, device=device)

    # 所有 token 索引
    token_ids = torch.arange(num_prefill_tokens, device=device)

    # 哪些是序列起始
    is_start = torch.zeros(num_prefill_tokens, dtype=torch.bool, device=device)
    is_start[query_start_loc[:-1]] = True

    # 找出非起始 token
    non_start_mask = ~is_start
    non_start_indices = non_start_mask.nonzero(as_tuple=True)[0]
    prev_indices = non_start_indices - 1

    # 拷贝前一个 hidden
    prev_hidden[non_start_indices] = hidden_states[prev_indices]

    return prev_hidden  # shape: [num_prefill_tokens, D]


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # self.gate_up_proj = MergedColumnParallelLinear(
        #     hidden_size,
        #     [intermediate_size] * 2,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.gate_up_proj",
        # )
        self.up_proj = RowParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.gate_proj = RowParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        
        rank = int(os.environ['SHIFT_VERSION'].split('-')[-1])
    
        self.shift_version = os.environ.get('SHIFT_VERSION', '').split('-')[0]

        if self.shift_version in ('v4cat', ):
            self.R = RowParallelLinear(
                2 * hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                intermediate_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                2 * hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v4cat_scale',):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                intermediate_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                2 * hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v4sub', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                intermediate_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v4pre', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                intermediate_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v4pre_glu', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                intermediate_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2cat', ):
            self.R = RowParallelLinear(
                2 * hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                2 * hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2cat_scale', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                2 * hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2cat_scale_glu_relu', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                2 * hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
            
        if self.shift_version in ('v2cat_glu', 'v2cat_glu_silu'):
            self.R = RowParallelLinear(
                2 * hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                2 * hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2sub', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2pre', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2pre_sae', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2pre_glu', 'v2pre_glu_silu'):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version in ('v2pre_glu_bias', ):
            self.R = RowParallelLinear(
                hidden_size,
                rank,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )
                
            self.W = RowParallelLinear(
                rank,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

            self.scale = RowParallelLinear(
                hidden_size,
                rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shift_weight",
            )

        if self.shift_version == "v2cat":
            self.get_final_gate = self.get_final_gate_v2cat
        elif self.shift_version == "v2cat_scale":
            self.get_final_gate = self.get_final_gate_v2cat_scale
        elif self.shift_version == "v2cat_scale_glu_relu":
            self.get_final_gate = self.get_final_gate_v2cat_scale_glu_relu
        elif self.shift_version == "v2cat_glu":
            self.get_final_gate = self.get_final_gate_v2cat_glu
        elif self.shift_version == "v2cat_glu_silu":
            self.get_final_gate = self.get_final_gate_v2cat_glu_silu
        elif self.shift_version == "v4cat":
            self.get_final_gate = self.get_final_gate_v4cat
        elif self.shift_version == "v2sub":
            self.get_final_gate = self.get_final_gate_v2sub
        elif self.shift_version == "v4sub":
            self.get_final_gate = self.get_final_gate_v4sub
        elif self.shift_version == "v2pre":
            self.get_final_gate = self.get_final_gate_v2pre
        elif self.shift_version == "v2pre_sae":
            self.get_final_gate = self.get_final_gate_v2pre_sae
        elif self.shift_version in ("v2pre_glu", 'v2pre_glu_bias'):
            self.get_final_gate = self.get_final_gate_v2pre_glu
        elif self.shift_version in ("v2pre_glu_silu",):
            self.get_final_gate = self.get_final_gate_v2pre_glu_silu
        elif self.shift_version == "v4pre_glu":
            self.get_final_gate = self.get_final_gate_v4pre_glu
        elif self.shift_version == "v4pre":
            self.get_final_gate = self.get_final_gate_v4pre
        elif self.shift_version in ('v4cat_scale', ):
            self.get_final_gate = self.get_final_gate_v4cat_scale
        else:
            raise NotImplementedError

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = nn.SiLU()

        self.last_x = None
        self.prefix = prefix

    def get_final_gate_v4cat(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        shift_gate = self.W(self.R(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])[0] * alpha
        ori_gate = self.gate_proj(hidden_states)[0]
        return ori_gate + shift_gate

    def get_final_gate_v4cat_scale(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        shift_gate = self.W(self.R(prev_hidden_states)[0])[0] * alpha
        ori_gate = self.gate_proj(hidden_states)[0]
        return ori_gate + shift_gate

    def get_final_gate_v2cat(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        conv_hidden_states = hidden_states + self.W(self.R(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])[0] * alpha
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v2cat_scale(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        conv_hidden_states = hidden_states + self.W(self.R(prev_hidden_states)[0])[0] * alpha
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v2cat_scale_glu_relu(self, hidden_states, prev_hidden_states):
        alpha = F.relu(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        conv_hidden_states = hidden_states + self.W(self.R(prev_hidden_states)[0] * alpha)[0]
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v2cat_glu(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        conv_hidden_states = hidden_states + self.W(self.R(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0] * alpha)[0]
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v2cat_glu_silu(self, hidden_states, prev_hidden_states):
        alpha = self.act_fn(self.scale(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0])
        conv_hidden_states = hidden_states + self.W(self.R(torch.concat([prev_hidden_states, hidden_states], dim=-1))[0] * alpha)[0]
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v4sub(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(hidden_states - prev_hidden_states)[0])
        shift_gate = self.W(self.R(hidden_states - prev_hidden_states)[0])[0] * alpha
        ori_gate = self.gate_proj(hidden_states)[0]
        return ori_gate + shift_gate

    def get_final_gate_v2sub(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(hidden_states - prev_hidden_states)[0])
        conv_hidden_states = hidden_states + self.W(self.R(hidden_states - prev_hidden_states)[0])[0] * alpha
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v4pre(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(prev_hidden_states)[0])
        shift_gate = self.W(self.R(prev_hidden_states)[0])[0] * alpha
        ori_gate = self.gate_proj(hidden_states)[0]
        return ori_gate + shift_gate

    def get_final_gate_v4pre_glu(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(prev_hidden_states)[0])
        shift_gate = self.W(self.R(prev_hidden_states)[0] * alpha )[0]
        ori_gate = self.gate_proj(hidden_states)[0]
        return ori_gate + shift_gate

    def get_final_gate_v2pre(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(prev_hidden_states)[0])
        conv_hidden_states = hidden_states + self.W(self.R(prev_hidden_states)[0])[0] * alpha
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v2pre_glu(self, hidden_states, prev_hidden_states):
        alpha = F.sigmoid(self.scale(prev_hidden_states)[0])
        conv_hidden_states = hidden_states + self.W(self.R(prev_hidden_states)[0] * alpha)[0]
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def get_final_gate_v2pre_glu_silu(self, hidden_states, prev_hidden_states):
        alpha = self.act_fn(self.scale(prev_hidden_states)[0])
        conv_hidden_states = hidden_states + self.W(self.R(prev_hidden_states)[0] * alpha)[0]
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate


    def get_final_gate_v2pre_sae(self, hidden_states, prev_hidden_states):
        conv_hidden_states = hidden_states + self.W(self.act_fn(self.R(prev_hidden_states)[0]))[0]
        final_gate = self.gate_proj(conv_hidden_states)[0]
        return final_gate

    def forward(self, hidden_states, attn_metadata, last_kv_cache, last_kv_indices):
        """
        hidden_states: [T, D], 通常为 flattened batch
        last_kv_cache: 缓存每个 request 上一个 token 的 hidden
        last_kv_indices: 当前 batch 中每个 request 对应的缓存 index
        attn_metadata: 提供 query_start_loc 区分 prefill/decode
        """
        
        T, D = hidden_states.shape
        input_states = hidden_states.clone()

        num_prefills = attn_metadata.num_prefills
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # ==== Prefill阶段处理 ====
        if num_prefills > 0:
            assert attn_metadata.query_start_loc is not None
            query_start_loc = attn_metadata.query_start_loc[:num_prefills + 1]

            prev_hidden_states = get_prev_hidden_with_zero_start(hidden_states, query_start_loc)
            
            # 存储每个 sequence 的最后一个 token hidden
            from_indices = query_start_loc[1:] - 1
            to_indices = last_kv_indices[:num_prefills]
            last_kv_cache[to_indices] = hidden_states[from_indices]
            
        # ==== Decode阶段处理 ====
        if attn_metadata.num_decode_tokens > 0:
            decode_hidden = hidden_states[num_prefill_tokens:].clone()
            last_token_indices = last_kv_indices[num_prefills:]  # shape [B_decode]
            prev_hidden_states = last_kv_cache[last_token_indices]

            last_kv_cache[last_token_indices] = decode_hidden

        final_gate = self.get_final_gate(hidden_states, prev_hidden_states)
        
        up, _ = self.up_proj(hidden_states)
        hidden_states = self.act_fn(final_gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        
        return hidden_states
    

class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        last_kv_cache: Optional[torch.Tensor],
        last_kv_indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        
        # changed
        hidden_states = self.mlp(hidden_states, 
            attn_metadata=attn_metadata,
            last_kv_cache=last_kv_cache,
            last_kv_indices=last_kv_indices
        )
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = {} is less than "
                             "`num_hidden_layers` = {}. Please open an issue "
                             "to discuss this feature.".format(
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        ## added
        self.model_config = vllm_config.model_config
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.last_kv_cache_manager: Optional[LastHiddenStateCacheManager] = None
        
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.last_kv_cache_manager is None:
            self.last_kv_cache_manager = LastHiddenStateCacheManager(
                self.model_config.dtype,
                input_ids.device,
                self.max_num_seqs,
                self.config,
            )
        #Ensure kwargs have request_ids_to_seq_ids and finished_requests_ids.
        if "seqlen_agnostic_capture_inputs" not in kwargs:
            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]
            last_kv_caches, last_kv_indices = self.last_kv_cache_manager.get_last_hidden_states(  # noqa: E501
                request_ids_to_seq_ids,
                finished_requests_ids,
            )
        else:
            last_kv_caches, last_kv_indices = kwargs[
                "seqlen_agnostic_capture_inputs"]
            
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
                last_kv_caches[i],
                last_kv_indices,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # ("gate_up_proj", "gate_proj", 0),
            # ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class ShiftQwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, HasInnerState):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        # "gate_up_proj": [
        #     "gate_proj",
        #     "up_proj",
        # ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        # "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds, 
                                   **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.model.last_kv_cache_manager.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.model.last_kv_cache_manager.get_seqlen_agnostic_capture_inputs(  # noqa: E501
            batch_size)
        
class Qwen2EmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        # "gate_up_proj": [
        #     "gate_proj",
        #     "up_proj",
        # ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        # "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # TODO: Replace this model class with as_embedding_model(
        # Qwen2ForCausalLM) after changing the default pooling method
        if pooler_config.pooling_type is None:
            logger.warning(
                "This embedding model will default to last-token pooling in "
                "an upcoming version. To avoid breaking changes, you should "
                "pass `--override-pooler-config '{\"pooling_type\": \"MEAN\"}'`"
                " explicitly.")

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_caches, attn_metadata,
                          intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)
