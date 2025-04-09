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

from smolbox.core.commons import AUTORESOLVE, resolve


class HfToONNXConverter:
    def __init__(
        self,
        model_path=AUTORESOLVE,
        output_model_path=AUTORESOLVE,
        task="text-generation",
        opset=14,
    ):
        
        #thow exception if output_dir exists already
        if os.path.exists(output_model_path):
            raise ValueError(f"Output directory {output_model_path} already exists!")
        
        self.model_path = resolve("model_path", model_path)
        self.output_model_path = resolve("output_model_path", output_model_path, write=True)
        print("output_model_path: ", self.output_model_path)
        
        self.task = task
        self.opset = opset

    def run(self):
        if not os.path.exists(self.output_model_path):
            os.makedirs(self.output_model_path)

        main_export(
            model_name_or_path=self.model_path,
            output=self.output_model_path,
            task=self.task,
            opset=self.opset,
            tokenizer=self.model_path,
        )

        return f"Export complete. ONNX model saved to: {self.output_model_path}"


if __name__ == "__main__":
    fire.Fire(HfToONNXConverter)
