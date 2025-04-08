# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "fire",
#   "torch",
# ]
# ///

import os
from transformers import AutoModel, AutoTokenizer
import fire

class HfZeroOutModel:
    def __init__(
        self,
        model_name="gpt2",
        output_dir="smolbox_zeroed_model",
        use_auth_token=False
    ):
        
        #throw exception if output_dir already exists
        if os.path.exists(output_dir):
            raise ValueError(f"Output directory {output_dir} already exists!")
        
        
        self.model_name_or_path = model_name
        self.output_dir = output_dir
        self.use_auth_token = use_auth_token

    def convert(self):
        print(f"Loading model: {self.model_name_or_path}")
        model = AutoModel.from_pretrained(self.model_name_or_path, use_auth_token=self.use_auth_token)

        print("Zeroing out model parameters...")
        for name, param in model.named_parameters():
            param.data.zero_()
            if param.grad is not None:
                param.grad.zero_()

        print(f"Saving zeroed-out model to: {self.output_dir}")
        model.save_pretrained(self.output_dir)

        print("Saving tokenizer as well...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_auth_token=self.use_auth_token)
        tokenizer.save_pretrained(self.output_dir)

        return f"Model zeroed and saved to {self.output_dir}"


if __name__ == "__main__":
    fire.Fire(HfZeroOutModel)
