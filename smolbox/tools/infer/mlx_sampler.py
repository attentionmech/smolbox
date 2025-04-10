# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx_lm",  # Apple MLX framework
#   "fire",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import os
import fire

from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.base_tool import BaseTool

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate  # Assuming you're using mlx-lm or similar wrapper

class MLXSampler(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        prompt="Once upon a time",
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        device="auto",  # MLX handles device automatically
    ):
        self.model_path = resolve("model_path", model_path)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def run(self):
        model, tokenizer = load(self.model_path)


        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=self.prompt,
        )

        return result

if __name__ == "__main__":
    fire.Fire(MLXSampler)
