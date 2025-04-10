# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "fire",
#   "torch",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import os
import re
import fire
import torch
from transformers import AutoModel, AutoTokenizer
from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.base_tool import BaseTool


class ModelParamEditor(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        output_model_path=AUTORESOLVE,
        reset_type="zero",  # options: 'zero', 'random', or 'expression'
        param_pattern=".*",  # regex pattern to match parameter names
        param_expression=None,  # Python expression or lambda function for custom param reset
    ):
        model_path = resolve("model_path", model_path)
        output_model_path = resolve("output_model_path", output_model_path, write=True)

        print(f"Model path: {model_path}")
        print(f"Output path: {output_model_path}")

        if os.path.exists(output_model_path) and os.listdir(output_model_path):
            print(f"WARNING: Output directory {output_model_path} is non-empty.")

        self.model_path = model_path
        self.output_path = output_model_path
        self.reset_type = reset_type.lower()
        self.param_pattern = param_pattern
        self.param_expression = param_expression

        # Automatically assume 'expression' if param_expression is provided
        if self.param_expression:
            self.reset_type = "expression"

    def _reset_parameters(self, model):
        print(
            f"Resetting parameters with type: {self.reset_type}, pattern: '{self.param_pattern}'"
        )
        pattern = re.compile(self.param_pattern)

        for name, param in model.named_parameters():
            if not pattern.match(name):
                continue

            if self.reset_type == "expression":
                # Apply the custom expression (should be a lambda or a valid Python expression)
                if callable(self.param_expression):
                    self.param_expression(param)
                else:
                    raise ValueError("param_expression should be a callable (e.g., a lambda function).")
            elif self.reset_type == "zero":
                param.data.zero_()
            elif self.reset_type == "random":
                if hasattr(param, "reset_parameters"):
                    param.reset_parameters()
                else:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown reset_type: {self.reset_type}")

            if param.grad is not None:
                param.grad.zero_()

    def run(self):
        print(f"Loading model: {self.model_path}")
        model = AutoModel.from_pretrained(self.model_path)

        self._reset_parameters(model)

        print(f"Saving model to: {self.output_path}")
        model.save_pretrained(self.output_path)

        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(self.output_path)

        return f"Model reset ({self.reset_type}) and saved to {self.output_path}"


if __name__ == "__main__":
    fire.Fire(ModelParamEditor)
    # Uncomment to run test directly
    # test_hf_model_resetter()
