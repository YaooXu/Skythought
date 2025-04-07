from setuptools import setup

setup(name='vllm_add_shift_models',
      version='0.1',
      packages=['vllm_add_shift_models'],
      entry_points={
          'vllm.general_plugins':
          ["register_dummy_model = vllm_add_shift_models:register"]
      })