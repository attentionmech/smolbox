# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "fire",
#   "transformers",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import copy

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smolbox.core.state_manager import AUTORESOLVE, resolve


# doesn't modify the model, just supports sample time tweaks


class ModelTweakSampler:
    def __init__(
        self,
        model_path=AUTORESOLVE,
        prompt="Once upon a time",
        deltas="",
        temperature=0.6,
        max_length=100,
    ):
        self.model_path = resolve("model_path", model_path)
        self.prompt = prompt
        self.deltas = deltas
        self.temperature = temperature
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = None
        self._tokenizer = None
        self._original_state_dict = None

    def _load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path).to(
            self.device
        )
        self._original_state_dict = copy.deepcopy(self._model.state_dict())

    def _parse_deltas(self, deltas_str):
        deltas = {}
        if deltas_str:
            for pair in deltas_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=")
                    deltas[k.strip()] = float(v.strip())
        return deltas

    def list(self):
        if self._model is None:
            self._load_model()

        lines = []
        for name, param in self._model.named_parameters():
            shape = tuple(param.shape)
            total = param.numel()
            lines.append(f"{name:<60} shape={shape:<20} total={total}")
        return "\n".join(lines)

    def _apply_deltas(self, deltas):
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in deltas:
                    param += deltas[name]

    def run(self):
        self._load_model()
        self._model.load_state_dict(copy.deepcopy(self._original_state_dict))
        self._apply_deltas(self._parse_deltas(self.deltas))

        inputs = self._tokenizer(self.prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self._model.generate(
            input_ids=inputs["input_ids"],
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=50,
            do_sample=True,
        )

        return self._tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    fire.Fire(ModelTweakSampler)
