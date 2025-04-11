# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "fire",
#   "transformers",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from smolbox.core.state_manager import resolve, AUTORESOLVE
from smolbox.core.tool_manager import BaseTool
import fire


class ModelImporter(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        device=None,
    ):
        # Resolve model_path, ensuring that it's provided
        self.model_path = resolve("model_path", model_path)
        if not self.model_path:
            raise ValueError("model_path must be provided.")

        # Set the device (default to CUDA if available, otherwise CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def download_or_load(self):
        # Check if the model is available locally
        if not os.path.exists(self.model_path):
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Move model to the correct device (cuda or cpu)
        model.to(self.device)
        return model, tokenizer

    def run(self):
        model, tokenizer = self.download_or_load()
        return True


if __name__ == "__main__":
    fire.Fire(ModelImporter)
