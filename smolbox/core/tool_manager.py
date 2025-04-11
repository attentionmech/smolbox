# smolbox/tools/base_tool.py
import sys
import subprocess
from smolbox.config import DEFAULT_TOOLS_DIR
from smolbox.core.state_manager import next_state, reset_state, set, get, init_state, print_state, list_models
import os


class BaseTool:
    def __init__(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the init() method.")

    def run(self):
        raise NotImplementedError("Subclasses must implement the run() method.")


def check_uv():
    """
    Check if the 'uv' CLI tool is installed and working.
    Exits the program if 'uv --version' fails.
    """
    try:
        subprocess.run(
            ["uv", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("Ensure uv is working properly! (uv --version) failed")
        sys.exit(1)


def contains_base_tool_subclass(script_path):
    """
    Naively check if the script file contains a subclass of BaseTool.
    Just a string search – avoids import or exec issues.
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            contents = f.read()
            return "BaseTool" in contents and "class" in contents
    except Exception as e:
        print(f"Error reading {script_path}: {e}")
        return False


def exec_tool(script: str, *args, tools_dir=DEFAULT_TOOLS_DIR, **kwargs):
    check_uv()

    script_path = os.path.join(tools_dir, f"{script}.py")
    if not os.path.exists(script_path):
        print(f"Tool not found: {script_path}")
        sys.exit(1)

    if not contains_base_tool_subclass(script_path):
        print("Error: Script does not appear to define a BaseTool subclass.")
        sys.exit(1)

    kwarg_flags = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", "--prerelease", "allow", script_path] + list(args) + kwarg_flags

    try:
        print("-" * 20)
        print(">> Executing tool:", " ".join(cmd))
        print("-" * 20)

        # Stream tool output directly to stdout/stderr
        subprocess.run(cmd, check=True)

        next_state()

        print(">> Tool execution successful.\n\n")

    except subprocess.CalledProcessError as e:
        print(f"Tool execution failed with code {e.returncode}")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)




# Internal commands that don't map to tool scripts
INTERNAL_COMMANDS = {
    "ls": lambda args: list_tools(DEFAULT_TOOLS_DIR),
    "reset": lambda args: reset_state(),
    "set": lambda args: set(args[0], args[1]),
    "get": lambda args: get(args[0], True),
    "init": lambda args: init_state(),
    "state": lambda args: print_state(),
    "lsmo": lambda args: list_models(),
    # "version": lambda: print("smolbox v0.1.0"),  # Add more as needed
}


def list_tools(tools_dir=DEFAULT_TOOLS_DIR):
    """
    Print all available tools in the tools directory and subdirectories.
    """
    print("> Available tools:")
    for root, _, files in os.walk(tools_dir):
        for fname in files:
            if fname.endswith(".py") and not fname.startswith("_"):
                rel_path = os.path.relpath(os.path.join(root, fname), tools_dir)
                print(f"  - {rel_path.removesuffix('.py')}")
