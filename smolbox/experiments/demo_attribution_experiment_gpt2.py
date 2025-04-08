# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "fire",
#   "transformers",
#   "pygame",
# ]
# ///

import pygame
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import time
import numpy as np
import fire

class TokenAttributionVisualizer:
    def __init__(self, model_name="gpt2", prompt="Once upon a time ", step_delay=0.1, width=500, height=360):
        self.model_name = model_name
        self.prompt = prompt
        self.step_delay = step_delay
        self.width = width
        self.height = height
        
        # Config constants
        self._font_size = 20
        self._margin = 10
        self._legend_height = 20
        
        self._model = None
        self._tokenizer = None
        self._screen = None
        self._font = None
        self._small_font = None
        self._running = True

    def _load_model(self):
        """Initialize tokenizer and model"""
        self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self._model.eval()

    def _init_pygame(self):
        """Initialize Pygame components"""
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Token Attribution Animation")
        self._font = pygame.font.SysFont("monospace", self._font_size)
        self._small_font = pygame.font.SysFont("monospace", 16)

    def _get_attention_attribution(self, input_ids):
        """Calculate last-layer attention attribution"""
        with torch.no_grad():
            outputs = self._model(input_ids, output_attentions=True)
            attn_weights = outputs.attentions[-1]  # [batch, heads, seq, seq]
            avg_attn = attn_weights.mean(dim=1)[0, -1]  # focus of last token
            return avg_attn.tolist()

    def _generate_next_token(self, input_ids):
        """Generate the next token"""
        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_token_id = torch.argmax(probs, dim=-1).item()
            return top_token_id

    def _wrap_tokens(self, tokens, colors, values, max_width):
        """Wrap tokens into lines for display"""
        lines = []
        current_line = []
        current_width = 150
        skip = len(tokens) - 40 if len(tokens) > 40 else 0

        for token, color, val in zip(tokens, colors, values):
            if skip > 0:
                skip -= 1
                continue
            rendered = self._font.render(token, True, color)
            token_width = rendered.get_width()
            if current_width + token_width > max_width:
                lines.append(current_line)
                current_line = [(token, color, val)]
                current_width = token_width + 150
            else:
                current_line.append((token, color, val))
                current_width += token_width
        if current_line:
            lines.append(current_line)
        return lines

    def _attribution_to_color(self, norm_val):
        """Convert attribution value to RGB color"""
        r = int(255 * norm_val)
        g = int(100 * (1 - norm_val))
        b = int(255 * (1 - norm_val))
        return (r, g, b)

    def _draw_legend(self):
        """Draw the color gradient legend"""
        for i in range(100):
            norm_val = i / 99
            color = self._attribution_to_color(norm_val)
            pygame.draw.rect(self._screen, color, 
                           pygame.Rect(self._margin + i*8, self.height - self._legend_height, 8, 20))
        min_label = self._small_font.render("Low Influence", True, (255, 255, 255))
        max_label = self._small_font.render("High Influence", True, (255, 255, 255))
        self._screen.blit(min_label, (self._margin, self.height - self._legend_height + 22))
        self._screen.blit(max_label, (self.width - self._margin - max_label.get_width(), 
                                    self.height - self._legend_height + 22))

    def _draw_screen(self, prompt_tokens, attributions):
        """Render the current state to the screen"""
        self._screen.fill((30, 30, 30))

        if attributions:
            max_val = max(attributions)
            min_val = min(attributions)
            norm_attr = [(a - min_val) / (max_val - min_val + 1e-6) for a in attributions]
            colors = [self._attribution_to_color(a) for a in norm_attr]
        else:
            colors = [(255, 255, 255)] * len(prompt_tokens)
            norm_attr = [0.0] * len(prompt_tokens)

        lines = self._wrap_tokens(prompt_tokens, colors, norm_attr, self.width - 2 * self._margin)
        y = self._margin
        for line in lines:
            x = self._margin
            for token, color, val in line:
                surf = self._font.render(token, True, color)
                self._screen.blit(surf, (x, y))
                val_surf = self._small_font.render(f"{val:.2f}", True, (200, 200, 200))
                self._screen.blit(val_surf, (x, y + self._font_size))
                x += max(surf.get_width(), val_surf.get_width()) + 10
            y += self._font_size + 25

        self._draw_legend()
        pygame.display.flip()

    def start(self):
        """Main execution loop"""
        self._load_model()
        self._init_pygame()
        
        prompt_ids = self._tokenizer.encode(self.prompt)
        tokens_so_far = self._tokenizer.convert_ids_to_tokens(prompt_ids)

        while self._running:
            input_ids = torch.tensor([prompt_ids])
            attributions = self._get_attention_attribution(input_ids) if len(prompt_ids) > 1 else []
            tokens_display = [self._tokenizer.convert_tokens_to_string([t]) for t in tokens_so_far]
            self._draw_screen(tokens_display, attributions)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            next_token_id = self._generate_next_token(input_ids)
            prompt_ids.append(next_token_id)
            next_token = self._tokenizer.convert_ids_to_tokens([next_token_id])[0]
            tokens_so_far.append(next_token)

            time.sleep(self.step_delay)

        pygame.quit()

if __name__ == "__main__":
    fire.Fire(TokenAttributionVisualizer)