# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "fire",
#   "transformers",
# ]
# ///

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire

class TokenBiasSampler:
    def __init__(self, model_name="gpt2", prompt="In a distant future,", temperature=0.7, max_length=100, bias_tokens="", bias_strength=5.0):
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.max_length = max_length
        self.bias_tokens = bias_tokens  # e.g. "robot, apocalypse, AI"
        self.bias_strength = bias_strength
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = None
        self._tokenizer = None
        self._bias_token_ids = []

    def _load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        if self.bias_tokens:
            tokens = [t.strip() for t in self.bias_tokens.split(",")]
            self._bias_token_ids = [self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(t))[0] for t in tokens if t]

    def _logits_processor(self):
        class TokenBiasProcessor:
            def __init__(self, token_ids, strength):
                self.token_ids = token_ids
                self.strength = strength

            def __call__(self, input_ids, scores):
                for tid in self.token_ids:
                    if 0 <= tid < scores.size(-1):
                        scores[:, tid] += self.strength
                return scores

        return TokenBiasProcessor(self._bias_token_ids, self.bias_strength)

    def sample(self):
        print("[NOTE] Current it's not working!")
        self._load_model()

        inputs = self._tokenizer(self.prompt, return_tensors="pt").to(self.device)

        logits_processor = self._logits_processor()

        # Set pad_token_id to eos_token_id if pad_token_id is None
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        output = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # add this
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True,
            top_k=50,
            logits_processor=[logits_processor],
            pad_token_id=pad_token_id,  # add this
        )

        return self._tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    fire.Fire(TokenBiasSampler)
