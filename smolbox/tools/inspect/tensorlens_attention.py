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


class AttentionTensorLens(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        prompt="Once upon a time " * 20,
        max_new_tokens=1,
        host="localhost",
        port=8000,
    ):
        self.model_path = resolve("model_path", model_path)
        self.text_input = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.host = host
        self.port = port

    def run(self):
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        model.eval()

        # Fix pad token issues
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Tokenize input and create attention mask
        inputs = tokenizer(self.text_input, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Step 1: Forward pass for prompt attentions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

        for layer_idx, layer_attention in enumerate(outputs.attentions):  # (1, heads, seq, seq)
            for head_idx in range(layer_attention.shape[1]):
                attn_matrix = layer_attention[0, head_idx].detach().cpu().numpy()
                trace(f"attn_prompt_layer{layer_idx}_head{head_idx}", attn_matrix)

        # Step 2: Generate token and trace attentions
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=False,
                output_scores=False,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # HuggingFace returns attentions as Tuple[Tuple[tensor]] for generate
        # We grab attentions from the last generated step (there's only one step here)
        gen_attn_step = gen_outputs.attentions[0]  # Tuple of tensors, one per layer

        for layer_idx, layer_attention in enumerate(gen_attn_step):
            for head_idx in range(layer_attention.shape[1]):
                attn_matrix = layer_attention[0, head_idx].detach().cpu().numpy()
                trace(f"attn_gen_layer{layer_idx}_head{head_idx}", attn_matrix)

        # Launch viewer
        viewer(height="100%", port=self.port, host=self.host)
        return True


if __name__ == "__main__":
    fire.Fire(AttentionTensorLens)
