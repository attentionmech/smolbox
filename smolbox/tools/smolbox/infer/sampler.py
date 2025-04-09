# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "torch",
#   "fire",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import os

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smolbox.core.state_manager import AUTORESOLVE, resolve

# plain old sampler

class BasicSampler:
    def __init__(
        self,
        model_path=AUTORESOLVE,
        prompt="Once upon a time",
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        device=None,
    ):
        self.model_path = resolve("model_path", model_path)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):

        # print(f"Loading model from: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()

        # print(f"Running inference on prompt: {self.prompt}")
        inputs = tokenizer(self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        # print("\n=== Output ===")
        # print(decoded)
        return decoded


if __name__ == "__main__":
    fire.Fire(BasicSampler)
