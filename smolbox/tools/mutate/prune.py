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
import torch
import fire
from transformers import AutoConfig, AutoModel, AutoTokenizer

from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.tools import BaseTool
from torch.nn.utils import prune


class ModelPruner(BaseTool):
    def __init__(
        self,
        model_path: str = AUTORESOLVE,
        output_model_path: str = AUTORESOLVE,
        pruning_percentage: float = 0.2,  # Percentage of weights to prune
        layer_type: str = "linear",  # Type of layers to prune (e.g., linear, conv)
    ):
        """
        Prune a model by removing a certain percentage of the smallest weights.

        Args:
            model_path: Path to the model to prune (local or HF hub)
            output_model_path: Path to save the pruned model
            pruning_percentage: Fraction of weights to prune (between 0 and 1)
            layer_type: Layer type to apply pruning (e.g., 'linear', 'conv')
        """
        model_path = resolve("model_path", model_path)
        output_model_path = resolve("output_model_path", output_model_path, write=True)

        print(f"Source model path: {model_path}")
        print(f"Output path: {output_model_path}")

        if os.path.exists(output_model_path) and os.listdir(output_model_path):
            print(f"WARNING: Output directory {output_model_path} is non-empty.")

        self.model_path = model_path
        self.output_path = output_model_path
        self.pruning_percentage = pruning_percentage
        self.layer_type = layer_type

    def _prune_layers(self, model):
        """Prune the specified layers of the model."""
        print(f"Pruning {self.layer_type} layers in the model.")

        # Loop through model layers and prune the ones that match the specified type
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and self.layer_type == "linear":
                print(f"Pruning {name}...")
                prune.l1_unstructured(
                    module, name="weight", amount=self.pruning_percentage
                )

            elif isinstance(module, torch.nn.Conv2d) and self.layer_type == "conv":
                print(f"Pruning {name}...")
                prune.l1_unstructured(
                    module, name="weight", amount=self.pruning_percentage
                )

        return model

    def run(self):
        """Prune the model and save the pruned version."""
        print(f"Loading model from: {self.model_path}")
        model = AutoModel.from_pretrained(self.model_path)

        print("Pruning model...")
        pruned_model = self._prune_layers(model)

        print(f"Saving pruned model to: {self.output_path}")
        pruned_model.save_pretrained(self.output_path)

        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(self.output_path)

        print(f"Model pruned and saved to {self.output_path}")

        return True


if __name__ == "__main__":
    fire.Fire(ModelPruner)
    # Example usage: python script.py --model_path="gpt2" --output_model_path="./pruned_gpt2" --pruning_percentage=0.3 --layer_type="linear"
