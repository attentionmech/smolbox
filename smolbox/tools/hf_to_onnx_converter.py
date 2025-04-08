# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "optimum[onnxruntime]",
#   "fire",
# ]
# ///

from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer, AutoModel
import fire
import os

class HFToONNXConverter:
    def __init__(
        self,
        model_name_or_path="gpt2",
        output_dir="onnx_model",
        task="text-generation",
        opset=14,
        use_auth_token=False
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.task = task
        self.opset = opset
        self.use_auth_token = use_auth_token

    def convert(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        main_export(
            model_name_or_path=self.model_name_or_path,
            output=self.output_dir,
            task=self.task,
            opset=self.opset,
            tokenizer=self.model_name_or_path,
            use_auth_token=self.use_auth_token
        )

        return f"Export complete. ONNX model saved to: {self.output_dir}"


if __name__ == "__main__":
    fire.Fire(HFToONNXConverter)
