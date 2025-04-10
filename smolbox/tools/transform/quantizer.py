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
import fire
import torch
from transformers import AutoModel, AutoTokenizer
from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.base_tool import BaseTool


class ModelQuantizer(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        output_model_path=AUTORESOLVE,
        precision=8,  # Target precision in bits (e.g., 8 or 16)
        layer_types=["Linear"],  # List of layer types to quantize (default: Linear layers)
    ):
        model_path = resolve("model_path", model_path)
        output_model_path = resolve("output_model_path", output_model_path, write=True)

        print(f"Model path: {model_path}")
        print(f"Output path: {output_model_path}")

        if os.path.exists(output_model_path) and os.listdir(output_model_path):
            print(f"WARNING: Output directory {output_model_path} is non-empty.")

        self.model_path = model_path
        self.output_path = output_model_path
        self.precision = precision
        self.layer_types = layer_types

    def _quantize_model(self, model):
        print(f"Quantizing model to {self.precision}-bit precision...")

        # Dynamic quantization typically supports linear layers and other types of layers
        # We will filter layers based on the user-defined `layer_types`
        quantized_model = model

        # Apply dynamic quantization to specified layers
        if self.precision == 8:
            print("Using 8-bit quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                quantized_model,
                {torch.nn.Linear},  # Apply quantization to Linear layers
                dtype=torch.qint8,  # 8-bit integer quantization
            )
        elif self.precision == 16:
            print("Using 16-bit quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                quantized_model,
                {torch.nn.Linear},  # Apply quantization to Linear layers
                dtype=torch.qint16,  # 16-bit integer quantization
            )
        else:
            raise ValueError(f"Unsupported precision: {self.precision}. Use 8 or 16.")

        return quantized_model

    def run(self):
        print(f"Loading model: {self.model_path}")
        model = AutoModel.from_pretrained(self.model_path)

        quantized_model = self._quantize_model(model)

        print(f"Saving quantized model to: {self.output_path}")
        quantized_model.save_pretrained(self.output_path)

        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(self.output_path)

        return f"Model quantized to {self.precision}-bit and saved to {self.output_path}"


if __name__ == "__main__":
    fire.Fire(ModelQuantizer)
    # Uncomment to run test directly
    # test_hf_model_quantizer()
