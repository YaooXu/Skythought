from vllm import LLM, SamplingParams

from vllm import ModelRegistry

import os


prompts = [
    "The president of the United States is",
]
sampling_params = SamplingParams(top_k=1)

os.environ['SHIFT_VERSION'] = 'v3.5'
llm = LLM(model="skythought/saves/math-short-cot-20k/Qwen2.5-7B-Instruct/lora-64-shift_gate/v3.5/complete_ckpt",
          enforce_eager=True)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")