import math
from typing import TYPE_CHECKING, Optional, Tuple

from numpy import dtype
import torch
import torch.nn as nn
import transformers
from transformers.models.qwen2.modeling_qwen2 import (
    Cache,
    Qwen2Model,
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    Qwen2ForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils.versions import require_version

from ...extras import logging
from ...extras.constants import SUPPORTED_CLASS_FOR_shift_gate
from ...extras.packages import is_transformers_version_greater_than
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import torch.nn.functional as F
from transformers.utils import is_flash_attn_greater_or_equal_2_10

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


transformers_logger = transformers.utils.logging.get_logger(__name__)

# class KVShiftingAttention(Attention):
#     def __init__(self, config):
#         super().__init__(config)

#         # Initialize KV shifting parameters
#         # Following paper's initialization: randomly initialize from U(0,1)
#         # and make them sum to 1
#         self.alpha1 = nn.Parameter(torch.rand(self.n_kv_head))
#         self.alpha2 = nn.Parameter(torch.ones(self.n_kv_head) - self.alpha1)
#         self.beta1 = nn.Parameter(torch.rand(self.n_kv_head))
#         self.beta2 = nn.Parameter(torch.ones(self.n_kv_head) - self.beta1)

#     def _shift_gate(self, x):
#         """Perform shifting operation on key/value tensors.
#         Shifts the sequence by padding a zero at the beginning and dropping last element.

#         Args:
#             x: Input tensor of shape (batch_size, seq_len, n_kv_head, head_dim)

#         Returns:
#             Shifted tensor of same shape
#         """
#         # Get shifted version by padding front and removing last element
#         # Keep same dimensions by dropping last element after padding
#         x_shifted = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
#         return x_shifted

#     def _project_kv(self, x, B, T):
#         """Override parent's _project_kv to add KV shifting.

#         Args:
#             x: Input tensor of shape (batch_size, seq_len, hidden_dim)
#             B: Batch size
#             T: Sequence length

#         Returns:
#             Tuple of processed key and value tensors
#         """
#         # Get initial K,V projections using parent method
#         kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
#         k, v = kv.unbind(dim=2)

#         # Get shifted versions
#         k_shifted = self._shift_gate(k)
#         v_shifted = self._shift_gate(v)

#         # Combine original and shifted versions with learned parameters
#         k = (
#             self.alpha1.view(1, 1, -1, 1) * k
#             + self.alpha2.view(1, 1, -1, 1) * k_shifted
#         )
#         v = self.beta1.view(1, 1, -1, 1) * v + self.beta2.view(1, 1, -1, 1) * v_shifted

#         return k, v

class KVShiftingAttention(nn.Module):
    def __init__(self, num_key_value_heads, dtype, device):
        super().__init__()
        # Initialize KV shifting parameters
        # Following paper's initialization: randomly initialize from U(0,1)
        # and make them sum to 1
        self.num_key_value_heads = num_key_value_heads
        
        self.alpha1 = nn.Parameter(torch.rand(num_key_value_heads, dtype=dtype, device=device))
        print(self.alpha1, self.alpha1.dtype, self.alpha1.device)
        # self.alpha2 = nn.Parameter(torch.zeros(self.num_key_value_heads))
        # self.beta1 = nn.Parameter(torch.ones(self.num_key_value_heads))
        # self.beta2 = nn.Parameter(torch.zeros(self.num_key_value_heads))
        
    def _shift_gate(self, x):
        """Perform shifting operation on key/value tensors.
        Shifts the sequence by padding a zero at the beginning and dropping last element.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_kv_head, head_dim)

        Returns:
            Shifted tensor of same shape
        """
        # Get shifted version by padding front and removing last element
        # Keep same dimensions by dropping last element after padding
        x_shifted = F.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        return x_shifted
    
    def forward(self, key_states, value_states):
        # (bsz, self.num_key_value_heads, q_len, self.head_dim)
        print(self.alpha1, self.alpha1.dtype)

        return self.alpha1.view(1, -1, 1, 1) * key_states, value_states
    
        
        dtype = key_states.dtype
        # Get shifted versions
        k_shifted = self._shift_gate(key_states)
        v_shifted = self._shift_gate(value_states)        
        # Combine original and shifted versions with learned parameters
        key_states = self.alpha1.view(1, -1, 1, 1) * key_states
        value_states = self.beta1.view(1, -1, 1, 1) * value_states
        
        return key_states.to(dtype), value_states.to(dtype)
    
def Qwen2Attention_post_init(self):
    self.shift_gate = KVShiftingAttention(self.num_key_value_heads, self.q_proj.weight.dtype, device=self.q_proj.weight.device)


def Qwen2FlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    ########## added ############
    # Get shifted versions
    key_states, value_states = self.shift_gate(key_states, value_states)
    #############################

    if position_embeddings is None:
        transformers_logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        transformers_logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _apply_qwen_patch() -> None:
    Qwen2FlashAttention2.forward = Qwen2FlashAttention2_forward
    Qwen2Attention.post_init = Qwen2Attention_post_init


def configure_shift_gate(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.shift_gate:
        return

    logger = logging.get_logger(__name__)

    if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_shift_gate:
        _apply_qwen_patch()
        logger.info_rank0("Using shift KV attention.")
    else:
        logger.warning_rank0("Current model does not support shift short attention.")
