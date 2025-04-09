# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "optimum[onnxruntime]",
#   "fire",
#   "smolbox@/Users/losh/focus/smolbox",
# ]
# ///

from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer, AutoModel
import fire
import os

class HfToONNXConverter:
    def __init__(
        self,
        model_path="gpt2",
        output_dir="smolbox_onnx_model",
        task="text-generation",
        opset=14,
    ):
        
        #thow exception if output_dir exists already
        if os.path.exists(output_dir):
            raise ValueError(f"Output directory {output_dir} already exists!")
        
        self.model_path = model_path
        self.output_dir = output_dir
        self.task = task
        self.opset = opset

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        main_export(
            model_name_or_path=self.model_path,
            output=self.output_dir,
            task=self.task,
            opset=self.opset,
            tokenizer=self.model_path,
        )

        return f"Export complete. ONNX model saved to: {self.output_dir}"


if __name__ == "__main__":
    fire.Fire(HfToONNXConverter)
