# smolbox

**smolbox** is a command-line toolkit designed for rapid experimentation and manipulation of language models. It provides a set of modular "tools" that operate on a managed state, allowing you to easily chain common LLM operations like importing, inspecting, modifying, training, and sampling.

Think of it as a workbench for your models. You load a model into the workspace, apply various tools to it, and the results are ready for the next step.

## Core Concepts

1.  **State Management:** `smolbox` maintains a simple state (stored in a hidden `.smolbox` directory) that tracks the current `model_path` and `dataset_path`. When a tool modifies a model or dataset, it typically saves the result to an `output_model_path` or `output_dataset_path`.
2.  **Tool Execution:** Tools are self-contained Python scripts. `smolbox` uses the `uv` build tool to execute them in isolated environments, automatically handling dependencies declared within each script file.
3.  **Workflow:** After a tool successfully runs, `smolbox` automatically updates the state: the `output_model_path` becomes the new `model_path`, and `output_dataset_path` becomes the new `dataset_path`. This makes it easy to chain operations together. You can often use `AUTORESOLVE` for path arguments, letting `smolbox` manage them automatically based on the current state.

## Prerequisites

*   **Python >= 3.11**
*   **uv:** The `uv` tool is required to run the tools. Install it via pipx (recommended) or pip:
    ```bash
    # Recommended
    pipx install uv
    # Or using pip
    pip install uv
    ```

## Installation

Install `smolbox` directly from the repository:

```bash
pip install git+https://github.com/attentionmech/smolbox.git
```

## Basic Usage

The general command structure is:

```bash
smolbox <tool_name> [tool_arguments...]
```

For example, to run the basic sampler tool:

```bash
smolbox infer/sample --prompt "The weather today is" --max_new_tokens 20
```

`smolbox` also has internal commands for managing state and listing tools:

*   `smolbox ls`: List all available tools.
*   `smolbox state`: Print the current state (model path, dataset path, etc.).
*   `smolbox init`: Initialize a new, empty state (clears previous state and outputs).
*   `smolbox reset`: Remove the `.smolbox` state directory entirely.
*   `smolbox set <key> <value>`: Manually set a state variable (e.g., `smolbox set model_path gpt2`).
*   `smolbox get <key>`: Print the value of a state variable.
*   `smolbox lsmo`: (Experimental) List models found in the internal state directory.

## Available Tools

Tools are organized into categories. Use `smolbox ls` to see the full list.

### `infer` - Generate text

*   **`infer/sample`**: Generate text samples from the current model using standard sampling parameters (temperature, top-k, top-p).
*   **`infer/tweak`**: Generate text while applying temporary modifications (deltas) to model parameters at sample time (doesn't save changes).

### `inspect` - Look inside the model

*   **`inspect/logitlens`**: Visualize the model's predictions at each layer during generation using the "logit lens" technique (requires `nnsight`). Provides a terminal-based animation.
*   **`inspect/mav`**: Run the Model Attribution Visualizer (MAV) to inspect token attributions during generation (requires `openmav`).
*   **`inspect/tensorlens_activations`**: Visualize model activations during a forward pass using TensorLens (requires `tensorlens`). Launches a web viewer.
*   **`inspect/tensorlens_attention`**: Visualize attention patterns using TensorLens (requires `tensorlens`). Launches a web viewer.
*   **`inspect/tensorlens_weights`**: Visualize the model's weight matrices using TensorLens (requires `tensorlens`). Launches a web viewer.

### `io` - Import and Export Models

*   **`io/import`**: Load a model from Hugging Face Hub or a local path into the `smolbox` state (essentially sets the `model_path`).
*   **`io/export`**: Save the current model in the state to a specified path and format (currently supports PyTorch `.pt` format).

### `mutate` - Modify Model Weights/Architecture

*   **`mutate/edit`**: Modify specific model parameters based on a pattern. Can zero-out, re-initialize randomly, or apply custom functions (currently only zero/random).
*   **`mutate/prune`**: Apply unstructured magnitude pruning to specified layer types (e.g., Linear layers).
*   **`mutate/quantize`**: Apply dynamic quantization (currently 8-bit or 16-bit) to specified layer types. *Note: Quantization support might be basic.*
*   **`mutate/reset`**: Create a new, randomly initialized model based on the configuration of the current model. Optionally change the number of layers.

### `train` - Train or Fine-tune Models

*   **`train/finetune`**: Fine-tune the current model on a specified dataset (Hugging Face dataset name or local path) using the Transformers `Trainer`.

## Example Workflow

Let's import a small model, fine-tune it on a tiny dataset, and then sample from it.

1.  **Initialize state:**
    ```bash
    smolbox init
    ```

2.  **Import the base model (e.g., GPT-2):**
    ```bash
    smolbox io/import --model_path gpt2
    # The state now has model_path="gpt2"
    smolbox state
    ```

3.  **Set the dataset path (e.g., a tiny Shakespeare dataset):**
    *You can use `smolbox set dataset_path tiny_shakespeare` or pass it directly to the fine-tuning tool.*
    ```bash
    smolbox set dataset_path tiny_shakespeare
    smolbox state
    ```

4.  **Fine-tune the model for one epoch:**
    *   `model_path` and `dataset_path` are automatically resolved from the state.
    *   The fine-tuned model will be saved to `output_model_path` within the `.smolbox` directory.
    ```bash
    smolbox train/finetune --num_train_epochs 1 --batch_size 4 --max_train_samples 1000
    # After this runs, next_state() moves the output path to the input path for the next tool
    smolbox state
    ```

5.  **Sample from the fine-tuned model:**
    *   The `model_path` now points to the fine-tuned model.
    ```bash
    smolbox infer/sample --prompt "To be or not to be" --max_new_tokens 50
    ```

## Tool Dependencies (`/// script` Header)

Each tool script (`.py` file in `smolbox/tools/`) contains a special header block like this:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers",
#   "torch",
#   "fire",
#   "smolbox@git+https://github.com/attentionmech/smolbox", # Dependency on smolbox itself
#   # Other tool-specific dependencies...
# ]
# ///
```

This header tells `uv` which Python version and packages are required to run the tool. `smolbox` uses `uv run` to automatically create a temporary environment and install these dependencies before executing the script.
