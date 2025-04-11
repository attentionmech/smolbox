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
from transformers import AutoConfig, AutoModel, AutoTokenizer

from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.tools import BaseTool


class ModelFromConfigInitializer(BaseTool):
    def __init__(
        self,
        model_path: str = AUTORESOLVE,
        output_model_path: str = AUTORESOLVE,
        num_layers: int = None,  # Optional: override number of layers
    ):
        """
        Create a new model instance from an existing model's configuration.

        Args:
            model_path: Path to the existing model (local or HF hub) to get config from
            output_model_path: Where to save the new model
            num_layers: Optional override for the number of layers
        """
        model_path = resolve("model_path", model_path)
        output_model_path = resolve("output_model_path", output_model_path, write=True)

        print(f"Source model path: {model_path}")
        print(f"Output path: {output_model_path}")

        if os.path.exists(output_model_path) and os.listdir(output_model_path):
            print(f"WARNING: Output directory {output_model_path} is non empty.")

        self.model_path = model_path
        self.output_path = output_model_path
        self.num_layers = num_layers

    def _create_new_model(self):
        """Create a new model instance from the source model's configuration."""
        print(f"Loading configuration from: {self.model_path}")
        config = AutoConfig.from_pretrained(self.model_path)

        # Optionally override the number of layers
        if self.num_layers is not None:
            if hasattr(config, "n_layer"):  # For GPT-style models
                print(
                    f"Overriding number of layers from {config.n_layer} to {self.num_layers}"
                )
                config.n_layer = self.num_layers
            elif hasattr(config, "num_hidden_layers"):  # For BERT-style models
                print(
                    f"Overriding number of layers from {config.num_hidden_layers} to {self.num_layers}"
                )
                config.num_hidden_layers = self.num_layers
            else:
                print(
                    "Warning: Config does not have a recognizable layer count attribute"
                )

        print("Creating new model instance with configuration:")
        print(f"  Model type: {config.model_type}")
        print(
            f"  Layers: {self.num_layers if self.num_layers is not None else 'unchanged'}"
        )
        print(
            f"  Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}"
        )

        # Create a new model instance with the config
        model = AutoModel.from_config(config)

        # Weights are automatically initialized according to the model's default scheme
        return model

    def run(self):
        """Create and save the new model instance."""
        model = self._create_new_model()

        print(f"Saving new model to: {self.output_path}")
        model.save_pretrained(self.output_path)

        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(self.output_path)

        print(f"New model instance created from config and saved to {self.output_path}")

        return True


if __name__ == "__main__":
    fire.Fire(ModelFromConfigInitializer)
    # Example usage: python script.py --model_path="gpt2" --output_model_path="./new_gpt2" --num_layers=2
