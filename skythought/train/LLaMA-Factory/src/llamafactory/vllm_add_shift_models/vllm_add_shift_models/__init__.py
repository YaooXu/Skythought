from vllm import ModelRegistry
import os

def register():
    # Test directly passing the model
    # from .qwen2_shift_v35 import ShiftQwen2ForCausalLM
    if os.environ.get('USE_EAGER', False):
        from .qwen2_shift_v3cat_scale_relu import ShiftQwen2ForCausalLM
    else:
        from .qwen2_shift import ShiftQwen2ForCausalLM

    from .llama_shift import ShiftLlamaForCausalLM

    print(f"{ShiftQwen2ForCausalLM.__module__}.{ShiftQwen2ForCausalLM.__qualname__}")
    ModelRegistry.register_model("ShiftQwen2ForCausalLM", ShiftQwen2ForCausalLM)

    print(f"{ShiftQwen2ForCausalLM.__module__}.{ShiftLlamaForCausalLM.__qualname__}")
    ModelRegistry.register_model("ShiftLlamaForCausalLM", ShiftLlamaForCausalLM)