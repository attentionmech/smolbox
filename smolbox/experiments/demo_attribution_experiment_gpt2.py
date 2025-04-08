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

# ----- Config -----
WIDTH, HEIGHT = 500, 360
FONT_SIZE = 20
MARGIN = 10
STEP_DELAY = 0.1  # seconds between tokens
LEGEND_HEIGHT = 20

# ----- Init Model -----
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ----- Init Pygame -----
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Token Attribution Animation")
font = pygame.font.SysFont("monospace", FONT_SIZE)
small_font = pygame.font.SysFont("monospace", 16)
clock = pygame.time.Clock()

# ----- Attribution Calculation (last-layer attention) -----
def get_attention_attribution(input_ids):
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attn_weights = outputs.attentions[-1]  # [batch, heads, seq, seq]
        avg_attn = attn_weights.mean(dim=1)[0, -1]  # focus of last token
        return avg_attn.tolist()

# ----- Get Next Token -----
def generate_next_token(input_ids):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_token_id = torch.argmax(probs, dim=-1).item()
        return top_token_id

# ----- Wrap Text Helper -----
def wrap_tokens(tokens, colors, values, max_width):
    lines = []
    current_line = []
    current_width = 150

    skip = len(tokens) - 40 if len(tokens) > 40 else 0

    for token, color, val in zip(tokens, colors, values):
        
        if skip > 0:
            skip -= 1
            continue
        
        rendered = font.render(token, True, color)
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

# ----- Color Mapping -----
def attribution_to_color(norm_val):
    r = int(255 * norm_val)
    g = int(100 * (1 - norm_val))
    b = int(255 * (1 - norm_val))
    return (r, g, b)

# ----- Draw Legend -----
def draw_legend():
    for i in range(100):
        norm_val = i / 99
        color = attribution_to_color(norm_val)
        pygame.draw.rect(screen, color, pygame.Rect(MARGIN + i*8, HEIGHT - LEGEND_HEIGHT, 8, 20))
    min_label = small_font.render("Low Influence", True, (255, 255, 255))
    max_label = small_font.render("High Influence", True, (255, 255, 255))
    screen.blit(min_label, (MARGIN, HEIGHT - LEGEND_HEIGHT + 22))
    screen.blit(max_label, (WIDTH - MARGIN - max_label.get_width(), HEIGHT - LEGEND_HEIGHT + 22))

# ----- UI Drawing -----
def draw_screen(prompt_tokens, attributions):
    screen.fill((30, 30, 30))

    if attributions:
        max_val = max(attributions)
        min_val = min(attributions)
        norm_attr = [(a - min_val) / (max_val - min_val + 1e-6) for a in attributions]
        colors = [attribution_to_color(a) for a in norm_attr]
    else:
        colors = [(255, 255, 255)] * len(prompt_tokens)
        norm_attr = [0.0] * len(prompt_tokens)

    lines = wrap_tokens(prompt_tokens, colors, norm_attr, WIDTH - 2 * MARGIN)
    y = MARGIN
    for line in lines:
        x = MARGIN
        for token, color, val in line:
            surf = font.render(token, True, color)
            screen.blit(surf, (x, y))
            val_surf = small_font.render(f"{val:.2f}", True, (200, 200, 200))
            screen.blit(val_surf, (x, y + FONT_SIZE))
            x += max(surf.get_width(), val_surf.get_width()) + 10
        y += FONT_SIZE + 25

    draw_legend()
    pygame.display.flip()

# ----- Main Loop (Auto Animation) -----
prompt = "Once upon a time "
prompt_ids = tokenizer.encode(prompt)

running = True
tokens_so_far = tokenizer.convert_ids_to_tokens(prompt_ids)

while running:
    input_ids = torch.tensor([prompt_ids])
    attributions = get_attention_attribution(input_ids) if len(prompt_ids) > 1 else []
    tokens_display = [tokenizer.convert_tokens_to_string([t]) for t in tokens_so_far]
    draw_screen(tokens_display, attributions)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    next_token_id = generate_next_token(input_ids)
    prompt_ids.append(next_token_id)
    next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]
    tokens_so_far.append(next_token)

    time.sleep(STEP_DELAY)

pygame.quit()