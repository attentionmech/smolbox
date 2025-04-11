# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fire",
#   "torch",
#   "openmav",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import fire
import torch

from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.tool_manager import BaseTool

from openmav.mav import MAV


class MAVRunner(BaseTool):
    def __init__(
        self,
        model_path=AUTORESOLVE,
        prompt="Once upon a time ",
        max_new_tokens=10,
        limit_chars=250,
        temp=0.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        aggregation="l2",
        refresh_rate=0.1,
        interactive=False,
        selected_panels=None,
        num_grid_rows=1,
        max_bar_length=50,
        device=None,
        scale="linear",
        backend="transformers",
        seed=42,
    ):
        self.model_path = resolve("model_path", model_path)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.limit_chars = limit_chars
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.aggregation = aggregation
        self.refresh_rate = refresh_rate
        self.interactive = interactive
        self.selected_panels = selected_panels
        self.num_grid_rows = num_grid_rows
        self.max_bar_length = max_bar_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.backend = backend
        self.seed = seed

    def run(self):
        print(f"Running MAV with model: {self.model}")
        mav_instance = MAV(
            model=self.model_path,
            prompt=self.prompt,
            max_new_tokens=self.max_new_tokens,
            limit_chars=self.limit_chars,
            temp=self.temp,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            aggregation=self.aggregation,
            refresh_rate=self.refresh_rate,
            interactive=self.interactive,
            selected_panels=self.selected_panels,
            num_grid_rows=self.num_grid_rows,
            max_bar_length=self.max_bar_length,
            device=self.device,
            scale=self.scale,
            backend=self.backend,
            seed=self.seed,
        )
        return True


if __name__ == "__main__":
    fire.Fire(MAVRunner)
