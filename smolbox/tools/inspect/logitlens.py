# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "nnsight",
#   "fire",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

#  lesswrong post about this logitlens technique which we are using nnsight library for: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

import os
import sys
import time
import torch
import fire

from nnsight import LanguageModel
from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.tool_manager import BaseTool


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_logit_lens_terminal(frames, delay=1.0, max_tokens_per_layer=5, prompt=""):
    for frame_idx, frame in enumerate(frames):
        try:
            clear_terminal()
            print(f"{prompt} ")
            last_token = frame['prompt'][len(frames[0]['prompt']):].strip() or "(none)"
            print(last_token + "\n")

            for i, tokens_row in enumerate(frame['layer_words']):
                import re
                token_str = " | ".join(tokens_row[-max_tokens_per_layer:]).replace("\n", " ")
                token_str = re.sub(r"[-_]+", " ", token_str)
                print(f"[LAYER {i}]: {token_str[-50:]}\n")

            sys.stdout.flush()
            time.sleep(delay)
        except Exception as e:
            print(f"[Error in frame {frame_idx}]: {str(e)}")


def get_layers_and_norm(model):
    """Detect and return the transformer layers and norm based on model architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Detected LLaMA-style model.")
        return model.model.layers, model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        print("Detected GPT-style model.")
        return model.transformer.h, model.transformer.ln_f
    else:
        raise ValueError("Unsupported model architecture: can't find transformer layers or norm.")


def compute_logit_lens(prompt, model):
    print(f"Computing logit lens for prompt: '{prompt}'")
    probs_layers = []

    layers, final_norm = get_layers_and_norm(model)

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer in layers:
                layer_out = model.lm_head(final_norm(layer.output[0]))
                probs = torch.nn.functional.softmax(layer_out, dim=-1).save()
                probs_layers.append(probs)

            input_ids = invoker.inputs[0][0]["input_ids"][0]

    probs = torch.cat([p.value.unsqueeze(0) for p in probs_layers], dim=0)
    max_probs, tokens = probs.max(dim=-1)

    last_n = 10
    input_words = [model.tokenizer.decode(t) for t in input_ids[-last_n:]]
    words = [
        [model.tokenizer.decode(t.cpu()) for t in layer_tokens[-last_n:]]
        for layer_tokens in tokens
    ]

    max_probs = max_probs[:, -last_n:]
    return max_probs.detach().cpu().numpy(), words, input_words


def autoregressive_logit_lens_animation(prompt, model, temperature=1.0, max_steps=5):
    frames = []
    current_prompt = prompt
    print(f"Starting generation with initial prompt: '{prompt}'")

    for step in range(max_steps):
        print(f"Generating step {step + 1}/{max_steps}...")
        max_probs, layer_words, input_words = compute_logit_lens(current_prompt, model)

        frames.append({
            'step': step,
            'prompt': current_prompt,
            'max_probs': max_probs,
            'layer_words': layer_words,
            'input_words': input_words,
        })

        inputs = model.tokenizer(current_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1]

        probs = torch.nn.functional.softmax(last_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        next_token = model.tokenizer.decode(next_token_id)

        if next_token.strip() == "":
            next_token = " " + next_token

        print(f"Generated token: '{next_token}'")
        current_prompt += next_token

    print(f"Final text: '{current_prompt}'")
    return frames


class LogitLensRunner(BaseTool):
    def __init__(
        self,
        prompt="The meaning of life is",
        temperature=1.0,
        max_tokens=5,
        tokens_per_layer=5,
        delay=1.0,
        debug=False,
        model_name="openai-community/gpt2",
        device_map="auto",
        dispatch=True,
    ):
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokens_per_layer = tokens_per_layer
        self.delay = delay
        self.debug = debug
        self.model_name = model_name
        self.device_map = device_map
        self.dispatch = dispatch

    def run(self):
        print("Loading model...")
        try:
            model = LanguageModel(self.model_name, device_map=self.device_map, dispatch=self.dispatch)
            print("Model loaded successfully!")

            frames = autoregressive_logit_lens_animation(
                prompt=self.prompt,
                model=model,
                temperature=self.temperature,
                max_steps=self.max_tokens
            )

            if not frames:
                print("Error: No frames generated!")
                return

            print(f"Generated {len(frames)} frames. Starting animation...")
            print_logit_lens_terminal(
                frames,
                delay=self.delay,
                max_tokens_per_layer=self.tokens_per_layer,
                prompt=self.prompt,
            )

            input("\n")

        except Exception as e:
            print(f"Error: {str(e)}")
            if self.debug:
                import traceback
                print(traceback.format_exc())
        return True


if __name__ == "__main__":
    def main(**kwargs):
        runner = LogitLensRunner(**kwargs)
        runner.run()

    fire.Fire(main)
