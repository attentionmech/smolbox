# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "optimum[onnxruntime]",
#   "fire",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import os
import torch
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import main_export
from smolbox.core.state_manager import resolve, AUTORESOLVE
from smolbox.core.tools import BaseTool


class ModelExporter(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        device=None,
        export_model_path="export_model",
        export_format="pt",  # Default to PyTorch format
    ):
        # Resolve model_path, ensuring that it's provided
        self.model_path = resolve("model_path", model_path)
        if not self.model_path:
            raise ValueError("model_path must be provided.")

        # Resolve export_model_path
        self.export_model_path = resolve("export_model_path", export_model_path)

        # Validate export_format
        self.export_format = export_format.lower()
        if self.export_format not in ["pt"]:
            raise ValueError("Invalid export format. Supported formats: 'pt', 'onnx'.")

        # Set the device (default to CUDA if available, otherwise CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def download_or_load(self):
        # Check if the model is available locally
        if not os.path.exists(self.model_path):
            print(f"Model not found locally at {self.model_path}. Downloading...")
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            print(f"Model found locally at {self.model_path}. Loading model...")
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Move model to the correct device (cuda or cpu)
        model.to(self.device)
        return model, tokenizer

    def export_model(self, model, tokenizer):
        # Ensure export path is provided
        if not self.export_model_path:
            raise ValueError(
                "export_model_path must be provided for exporting the model."
            )

        if self.export_format == "pt":
            # Save the PyTorch model
            print(
                f"Exporting model to {self.export_model_path} as a PyTorch checkpoint..."
            )
            model.save_pretrained(self.export_model_path)
            tokenizer.save_pretrained(self.export_model_path)
        else:
            print("Not supported yet..")

    def run(self):
        model, tokenizer = self.download_or_load()
        print(f"Model loaded successfully on device: {self.device}")

        if self.export_model_path:
            self.export_model(model, tokenizer)
            print(
                f"Model exported to {self.export_model_path} in {self.export_format} format."
            )
        else:
            print("No export path provided. Not exporting the model.")
            return False

        return True


if __name__ == "__main__":
    fire.Fire(ModelExporter)
