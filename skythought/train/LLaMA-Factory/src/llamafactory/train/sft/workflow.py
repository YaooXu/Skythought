# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from collections import OrderedDict, defaultdict
import json
import re
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
import os
from peft import PeftModel
from transformers import Seq2SeqTrainingArguments, TrainerCallback

if TYPE_CHECKING:
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


class ABChangeWithGradCallback(TrainerCallback):
    def __init__(self, interval=20):
        self.interval = interval
        self.pattern = re.compile(r'layers\.(\d+)\.(self_attn|mlp)\.(\w+).*\.weight$')
        
        # 使用OrderedDict保证顺序
        self.grad_norms_A = OrderedDict()
        self.grad_norms_B = OrderedDict()
        self.handles = []

    def _extract_key(self, name):
        match = self.pattern.search(name)
        if not match:
            return None
        return (int(match.group(1)), match.group(2), match.group(3))

    def _register_grad_hooks(self, model):
        """确保正确注册hook并初始化存储"""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            key = self._extract_key(name)
            if not key:
                continue
                
            if 'lora_A' in name:
                # 初始化存储
                self.grad_norms_A[key] = []
                # 注册hook（使用闭包保存当前key）
                def make_hook_A(k):
                    def hook(grad):
                        if grad is not None:
                            self.grad_norms_A[k].append(torch.norm(grad.float()).item())
                    return hook
                self.handles.append(param.register_hook(make_hook_A(key)))
                
            elif 'lora_B' in name:
                self.grad_norms_B[key] = []
                def make_hook_B(k):
                    def hook(grad):
                        if grad is not None:
                            self.grad_norms_B[k].append(torch.norm(grad.float()).item())
                    return hook
                self.handles.append(param.register_hook(make_hook_B(key)))

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model:
            self._register_grad_hooks(model)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0:
            model = kwargs.get('model')
            if not model:
                return

            stats = {
                "global_step": state.global_step,
                "changes": OrderedDict(),
                "gradients_A": OrderedDict(),
                "gradients_B": OrderedDict()
            }

            # 获取参数组
            lora_As, lora_Bs, bases = {}, {}, {}
            for name, param in model.named_parameters():
                key = self._extract_key(name)
                if not key:
                    continue
                
                if 'lora_A' in name:
                    lora_As[key] = param.detach()
                elif 'lora_B' in name:
                    lora_Bs[key] = param.detach()
                elif 'base_layer' in name:
                    bases[key] = param.detach()

            # 计算统计量
            for key in lora_As:
                layer, module, proj = key
                proj_key = f"{module}.{proj}"
                
                base_weight = bases[key].float()
                
                # 参数变化
                BA = 2 * lora_Bs[key].float() @ lora_As[key].float()
                stats["changes"][proj_key] = (
                    torch.norm(BA) / (torch.norm(base_weight) + 1e-12)
                )
                
                # 梯度统计A
                if key in self.grad_norms_A and self.grad_norms_A[key]:
                    grads_A = self.grad_norms_A[key][-self.interval:]
                    stats["gradients_A"][proj_key] = {
                        "avg_norm": np.mean(grads_A),
                        "relative_norm": np.mean(grads_A) / (torch.norm(base_weight) + 1e-12)
                    }
                
                # 梯度统计B
                if key in self.grad_norms_B and self.grad_norms_B[key]:
                    grads_B = self.grad_norms_B[key][-self.interval:]
                    stats["gradients_B"][proj_key] = {
                        "avg_norm": np.mean(grads_B),
                        "relative_norm": np.mean(grads_B) / (torch.norm(base_weight) + 1e-12)
                    }

            # 清空当前梯度记录
            for k in self.grad_norms_A:
                self.grad_norms_A[k].clear()
            for k in self.grad_norms_B:
                self.grad_norms_B[k].clear()

            # 写入文件
            if args.local_rank in {-1, 0}:
                with open(f"{args.output_dir}/lora_stats.jsonl", "a") as f:
                    f.write(json.dumps(stats, default=float) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self.handles:
            handle.remove()
            
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    if model_args.shift_gate:
        assert 'SHIFT_VERSION' in os.environ
        training_args.output_dir = training_args.output_dir + '/' + os.environ['SHIFT_VERSION']
        training_args.run_name = training_args.run_name + '/' + os.environ['SHIFT_VERSION']
    
    if 'CHECKPOINT_SAVE' in os.environ:
        training_args.output_dir = os.path.join(os.environ['CHECKPOINT_SAVE'], training_args.output_dir)
        
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    if finetuning_args.finetuning_type == "lora" and data_args.dataset == 'Open-Thoughts-8k-10k':
        callbacks = [ABChangeWithGradCallback(interval=1)]
    else:
        pass
        
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if trainer.accelerator.is_local_main_process:
        if finetuning_args.finetuning_type.startswith("lora"):
            if finetuning_args.finetuning_type == "lora-ga":
                from peft.utils.lora_ga_utils import save_loraga_model_final
                peft_dir = os.path.join(trainer.args.output_dir, "final_lora_ckpt")
                model = trainer.accelerator.unwrap_model(trainer.model)
                save_loraga_model_final(model, save_dir=peft_dir)
            else:
                peft_dir = os.path.join(trainer.args.output_dir, f"{'checkpoint'}-{trainer.state.global_step}")
            
            del model
            
            new_model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, load_bare_model=True)
            new_model = PeftModel.from_pretrained(new_model, peft_dir)
            
            print(new_model.device)
            print('merging')
            new_model = new_model.merge_and_unload()
            
            if model_args.infer_dtype == "auto":
                output_dtype = getattr(new_model.config, "torch_dtype", torch.float16)
            else:
                output_dtype = getattr(torch, model_args.infer_dtype)

            print(output_dtype)
            setattr(new_model.config, "torch_dtype", output_dtype)
            new_model = new_model.to(output_dtype)
                            
            final_dir = os.path.join(trainer.args.output_dir, "complete_ckpt")
            
            new_model.save_pretrained(final_dir, max_shard_size='4GB', safe_serialization=True)
            tokenizer.save_pretrained(final_dir)

        else:
            final_dir = trainer.args.output_dir
            
        if model_args.shift_gate:
            config_path = os.path.join(final_dir, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            config['architectures'][0] = 'Shift' + config['architectures'][0]
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)
        
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
