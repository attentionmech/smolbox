# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "fire",
#   "transformers",
# ]
# ///

import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire

class ParamTweakSampler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.original_state_dict = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.original_state_dict = copy.deepcopy(self.model.state_dict())

    def parse_deltas(self, deltas_str):
        deltas = {}
        if deltas_str:
            for pair in deltas_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=")
                    deltas[k.strip()] = float(v.strip())
        return deltas

    def apply_deltas(self, deltas):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in deltas:
                    param += deltas[name]

    def run(self, prompt, deltas="", temperature=0.6, max_length=100):
        self.load_model()
        self.model.load_state_dict(copy.deepcopy(self.original_state_dict))
        self.apply_deltas(self.parse_deltas(deltas))
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            do_sample=True
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

def run_fn(
    prompt="Once upon a time",
    deltas="",
    temperature=0.6,
    max_length=100,
    model_name="gpt2",
):
    """
    Generate text from a language model with optional weight deltas applied.

    Args:
        prompt (str): The input prompt to generate from.
        deltas (str): Comma-separated layer deltas (e.g., "layer.name=0.01").
        temperature (float): Sampling temperature.
        max_length (int): Maximum tokens to generate.
        model_name (str): HuggingFace model ID (e.g., gpt2, mistralai/Mistral-7B-v0.1).
    """
    runner = ParamTweakSampler(model_name)
    result = runner.run(prompt=prompt, deltas=deltas, temperature=temperature, max_length=max_length)

    print("\n> Generated Text:\n")
    print(result)
    return result

if __name__ == "__main__":
    fire.Fire(run_fn)

