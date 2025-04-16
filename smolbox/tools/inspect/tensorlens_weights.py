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

import numpy as np
from transformers import AutoModel, AutoConfig
from tensorlens.tensorlens import trace, viewer


class TensorLensWeights(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
    ):
        self.model_path = resolve("model_path", model_path)

    def run(self):
        
        model_name = self.model_path

        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        [trace(key, tensor.detach().cpu().numpy()) for key, tensor in model.state_dict().items()]

        viewer(height='100%', port=8000)

        return True


if __name__ == "__main__":
    fire.Fire(TensorLensWeights)
