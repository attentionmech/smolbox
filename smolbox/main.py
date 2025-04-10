import os
import subprocess
import sys

import fire
from smolbox.core.state_manager import next_state, reset_state, set, get, init_state,print_state

# Default directories (can be overridden via CLI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOOLS_DIR = os.path.join(BASE_DIR, "tools")


# Internal commands that don't map to tool scripts
INTERNAL_COMMANDS = {
    "ls": lambda args: list_tools(DEFAULT_TOOLS_DIR),
    "reset": lambda args: reset_state(),
    "set": lambda args: set(args[0], args[1]),
    "get": lambda args: get(args[0], True),
    "init": lambda args: init_state(),
    "state": lambda args: print_state(),
    # "version": lambda: print("smolbox v0.1.0"),  # Add more as needed
}


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
    """
    Run a tool script from the tools directory using 'uv run'.
    """
    check_uv()

    script_path = os.path.join(tools_dir, f"{script}.py")
    if not os.path.exists(script_path):
        print(f"Tool not found: {script_path}")
        sys.exit(1)

    if not contains_base_tool_subclass(script_path):
        print("Error: Script does not appear to define a BaseTool subclass.")
        sys.exit(1)

    kwarg_flags = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", script_path] + list(args) + kwarg_flags

    try:
        print("-"*20)
        print(">> Executing tool:", " ".join(cmd))
        print("-"*20)
        
        subprocess.run(cmd, check=True)
        next_state()
        print(">> Tool execution successful.\n\n")
        print("\n\n\n")
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
    if len(sys.argv) < 2:
        print("Usage: smolbox <toolname> [args]  — or —  smolbox " + " | ".join(INTERNAL_COMMANDS.keys()))
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]
    if command in INTERNAL_COMMANDS:
        INTERNAL_COMMANDS[command](args)
        return

    script_path = os.path.join(DEFAULT_TOOLS_DIR, f"{command}.py")
    if "/" not in command and not os.path.exists(script_path):
        print(f"Unknown command or tool: '{command}'")
        print("Run `smolbox ls` to see available tools.")
        sys.exit(1)

    if not args or args[0] not in ("run", "help", "--help", "-h"):
        args = ["run"] + args

    exec_tool(command, *args)



if __name__ == "__main__":
    main()
