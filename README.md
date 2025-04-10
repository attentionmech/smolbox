# smolbox

**smolbox** is a simple toolbox (underconstruction) for basic LLM operations with opinionated state management between the tools to speed up user workflow.

It leverages `uv` to run each tool in an isolated environment with its specific dependencies, ensuring tools don't interfere with each other.

The power lies in chaining tools, for example below is a shell script which runs smolbox to do a sequence of operations:

```
smox init

smox io/importer --model-path gpt2

smox infer/sampler --prompt "thou shall haveth"

smox transform/reinitializer --num_layers 4 

smox train/finetuner --dataset-path karpathy/tiny_shakespeare --num-train-epochs 100

smox infer/sampler --prompt "thou shall haveth"

smox transform/pruner

smox infer/sampler

smox transform/quantizer

smox infer/sampler

smox io/exporter
```


## Core Concepts

1.  **Tools:** Self-contained Python scripts located in the `smolbox/tools/` directory. Each tool performs a specific ML operation (e.g., `infer/sampler`, `train/finetuner`). Tools inherit from `BaseTool`.
2.  **State Management:** A simple state manager (`smolbox/core/state_manager.py`) keeps track of crucial paths (`model_path`, `dataset_path`, `output_model_path`, etc.) in a `.smolbox` directory within your current working directory.
3.  **Chaining & `AUTORESOLVE`:** Tools can be run sequentially. The state manager facilitates this by automatically transitioning output paths (`output_model_path`) to become input paths (`model_path`) for the next step (`next_state()` function). Tools often use `AUTORESOLVE` as a default value for paths, allowing them to automatically pick up the relevant path from the current state.
4.  **Dependency Isolation:** Each tool defines its dependencies in a `/// script ... ///` header block. The main `smolbox` command uses `uv run` to execute the tool, automatically setting up a virtual environment and installing the required packages for that specific run.

## Features

*   **Modular:** Easily add new tools by creating Python scripts inheriting from `BaseTool`.
*   **Stateful:** Tracks model and dataset paths across commands.
*   **Chaining:** Output of one tool can seamlessly become the input for the next.
*   **Dependency Management:** Uses `uv` for isolated, per-tool dependency handling.
*   **Common ML Tasks:** Includes tools for inference, I/O, training, and model transformation (pruning, quantization, editing, reinitialization).
*   **Simple CLI:** Easy-to-use command-line interface powered by `fire`.
*   **Automatic Path Resolution:** `AUTORESOLVE` simplifies workflows by using state-managed paths automatically.

## Installation

1. clone and do local pip install via `pip install .`
2. Or can use `uv sync`

use via `smox` or `smolbox` cli commands

## Usage

### Basic Structure

```bash
smolbox <tool_name> [run] [tool_arguments...]
```

*   `<tool_name>`: The path to the tool relative to the `smolbox/tools` directory (e.g., `io/importer`, `infer/sampler`).
*   `[run]`: The `run` subcommand is implicitly assumed if not provided, executing the tool's `run` method. You can often omit it.
*   `[tool_arguments...]`: Arguments specific to the tool (e.g., `--prompt "Hello"`, `--num_train_epochs 1`). Use `--help` for tool-specific options: `smolbox <tool_name> --help`.

### Internal Commands

These commands manage `smolbox` itself and its state:

*   `smolbox ls`: List all available tools.
*   `smolbox init`: Initialize or re-initialize the state (clears the `.smolbox` directory).
*   `smolbox reset`: Clear the state and the `.smolbox` directory.
*   `smolbox state`: Print the current state (`state.json`).
*   `smolbox set <key> <value>`: Manually set a state variable (e.g., `smolbox set model_path gpt2`).
*   `smolbox get <key>`: Get and print a state variable.

### Example Workflow: Finetune & Sample

This example shows how to initialize state, set an initial model and dataset, finetune the model, and then sample from the finetuned version.

1.  **Initialize State:** (Start fresh in your project directory)
    ```bash
    smolbox init
    ```

2.  **Set Initial Model:** (Using a Hugging Face model ID)
    ```bash
    smolbox set model_path gpt2
    ```

3.  **(Optional) Verify Import:**
    ```bash
    smolbox io/importer
    # Tool execution successful.
    ```

4.  **Set Dataset:** (Using a Hugging Face dataset ID)
    ```bash
    smolbox set dataset_path benutzer/uncensored-creative-stories # Replace with a suitable dataset
    ```

5.  **Check State:**
    ```bash
    smolbox state
    # {
    #   "created_at": "...",
    #   "dataset_path": "benutzer/uncensored-creative-stories",
    #   "model_path": "gpt2",
    #   "output_dataset_path": null,
    #   "output_model_path": null,
    #   "updated_at": "..."
    # }
    ```

6.  **Finetune:** (Run for a short time; output model path defaults to `AUTORESOLVE`)
    ```bash
    smolbox train/finetuner --num_train_epochs 1 --batch_size 1
    # >> Executing tool: uv run smolbox/tools/train/finetuner.py run --num_train_epochs 1 --batch_size 1
    # ... (Training logs) ...
    # Model fine-tuned and saved to .smolbox/<uuid>
    # >> Tool execution successful.
    ```

7.  **Check State After Finetuning:** Notice `model_path` is now the output path from the finetuning step, and `output_model_path` is cleared, ready for the next tool.
    ```bash
    smolbox state
    # {
    #   "created_at": "...",
    #   "dataset_path": "benutzer/uncensored-creative-stories",
    #   "model_path": ".smolbox/<uuid>", # <-- Path to the finetuned model
    #   "output_dataset_path": null,
    #   "output_model_path": null,      # <-- Ready for next output
    #   "updated_at": "..."
    # }
    ```

8.  **Sample from Finetuned Model:** (Uses the `model_path` from the current state automatically because the tool's `model_path` defaults to `AUTORESOLVE`)
    ```bash
    smolbox infer/sampler --prompt "Once upon a time" --max_new_tokens 20
    # >> Executing tool: uv run smolbox/tools/infer/sampler.py run --prompt "Once upon a time" --max_new_tokens 20
    # ...
    # === Output ===
    # Once upon a time... (generated text) ...
    # >> Tool execution successful.
    ```

## Available Tools

Use `smolbox ls` to see the full list. Tools are organized into categories:

*   **`core`**: Base classes and state management (not directly executable as tools).
*   **`infer`**:
    *   `sampler`: Basic text generation/sampling from a model.
    *   `tweaker`: Sample from a model with temporary parameter tweaks (deltas).
*   **`io`**:
    *   `exporter`: Export a model (currently supports saving as PyTorch checkpoint `.pt`).
    *   `importer`: Import/load a model (useful for verification or starting a chain).
*   **`train`**:
    *   `finetuner`: Fine-tune a model on a given dataset.
*   **`transform`**:
    *   `editor`: Edit model parameters (zero out, randomize, apply expression).
    *   `pruner`: Apply weight pruning (L1 unstructured) to specified layer types.
    *   `quantizer`: Apply dynamic quantization (8-bit or 16-bit) to specified layer types.
    *   `reinitializer`: Create a new model instance using the configuration of an existing model (optionally changing layer count).

*For details on a specific tool's arguments, run:*
```bash
smolbox <tool_name> --help
```

## State Management Details

*   State is stored in `.smolbox/state.json` in your current working directory.
*   A history of states is appended to `.smolbox/state_history.jsonl`.
*   Key state variables managed:
    *   `model_path`: Path to the current input model.
    *   `output_model_path`: Path where the next tool should save its output model. Automatically generated if `AUTORESOLVE` is used for an output path.
    *   `dataset_path`: Path to the current input dataset.
    *   `output_dataset_path`: Path for output datasets (if applicable).
*   The `next_state()` function (called after successful tool execution) transitions the state:
    *   `model_path` becomes the previous `output_model_path`.
    *   `output_model_path` is reset to `None`.
    *   (Similar logic applies to `dataset_path` and `output_dataset_path`).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## NOTE

This is not for production, but mostly for learning and tinkering locally right now. Do not assume otherwise.
