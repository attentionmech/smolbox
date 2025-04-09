import os
import shutil
import subprocess
import sys

import fire

from smolbox.core.state_manager import next_state

# Default directories (can be overridden via CLI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOOLS_DIR = os.path.join(BASE_DIR, "tools")

def check_uv():
    """
    Check if the 'uv' CLI tool is installed and working.
    Exits the program if 'uv --version' fails.
    """
    try:
        _ = subprocess.run(
            ["uv", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("Ensure uv is working properly! (uv --version) failed")
        sys.exit(1)


def exec_tool(script: str, *args, tools_dir=DEFAULT_TOOLS_DIR, **kwargs):
    """
    Run a tool script from the tools directory using 'uv run'.

    Args:
        script (str): Path to script inside tools_dir, without '.py'.
                      e.g., 'subdir/myscript' maps to tools_dir/subdir/myscript.py
        args: Positional CLI arguments.
        tools_dir (str): Override tools directory.
        kwargs: Keyword arguments passed as CLI flags (--key=value).
    """
    check_uv()

    script_path = os.path.join(tools_dir, f"{script}.py")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    kwarg_flags = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", script_path] + list(args) + kwarg_flags

    try:
        subprocess.run(cmd, check=True)
        next_state()
    except subprocess.CalledProcessError as e:
        print(f"Tool execution failed with code {e.returncode}")
        sys.exit(1)



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

def main():
    fire.Fire(
        {
            "run": exec_tool,
            "ls": list_tools,
        }
    )


if __name__ == "__main__":
    main()
