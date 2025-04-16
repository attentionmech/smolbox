# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire",
#   "torch",
#   "tensorlens",
#   "transformers",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import fire
import torch
from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.tool_manager import BaseTool
from transformers import AutoTokenizer, AutoModelForCausalLM

from tensorlens.tensorlens import trace, viewer


class TensorLensActivations(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        prompt="Once upon a time "*10,
        max_new_tokens=1,  # Generate only 1 new token
        host="localhost",
        port=8000,
        notebook=False,
    ):
        self.model_path = resolve("model_path", model_path)
        self.text_input = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.host = host
        self.port = port
        self.notebook = notebook

    def run(self):
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        model.eval()

        # Set padding token if it's None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Tokenize the input text
        input_ids = tokenizer(
            self.text_input or "Once upon a time"*10,
            return_tensors="pt"
        ).input_ids

        # Step 1: Perform a forward pass to get activations for the prompt
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        # Trace activations for the prompt tokens (only)
        for step_idx, layer_outputs in enumerate(outputs.hidden_states):
            for layer_idx, hidden in enumerate(layer_outputs):
                trace(f"layer_{layer_idx}_step_{step_idx}", hidden.detach().cpu().numpy())

        # Step 2: Generate 1 new token based on the prompt
        with torch.no_grad():
            outputs_gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,  # Only generate 1 new token
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=False,
                output_attentions=False,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
            )

        # Step 3: Trace activations for the first generated token
        for step_idx, layer_outputs in enumerate(outputs_gen.hidden_states):
            for layer_idx, hidden in enumerate(layer_outputs):
                trace(f"layer_{layer_idx}_step_gen_{step_idx}", hidden.detach().cpu().numpy())

        viewer(height="100%", port=self.port, host=self.host, notebook=self.notebook)
        return True


if __name__ == "__main__":
    fire.Fire(TensorLensActivations)
