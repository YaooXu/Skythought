from vllm import ModelRegistry


def register():
    # Test directly passing the model
    # from .qwen2_shift_v35 import ShiftQwen2ForCausalLM
    from .qwen2_shift import ShiftQwen2ForCausalLM
    # from .qwen2_shift_v2cat_scale_relu import ShiftQwen2ForCausalLM

    if "Qwen2ShiftForCausalLM" not in ModelRegistry.get_supported_archs():
        print(f"{ShiftQwen2ForCausalLM.__module__}.{ShiftQwen2ForCausalLM.__qualname__}")
        ModelRegistry.register_model("ShiftQwen2ForCausalLM", ShiftQwen2ForCausalLM)