import os
import sys
import fire
import shutil
import subprocess


TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")

def check_uv():
    try:
        _ = subprocess.run(["uv", "--version"], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        print("Ensure uv is working properly! (uv --version) failed")
        sys.exit(1)


import os
import subprocess

# Update TOOLS_DIR to match your project layout
TOOLS_DIR = "/Users/losh/focus/smolbox/smolbox/tools"

def check_uv():
    # Dummy check — replace with actual availability test if needed
    if not shutil.which("uv"):
        raise RuntimeError("uv is not installed or not in PATH")

def exec_tool(script: str, *args, **kwargs):
    """
    Run a smolbox tool script using uv run.

    Args:
        script (str): Script name (e.g., param_tweak_sampler).
        args: Positional CLI arguments (e.g., subcommands like 'sample').
        kwargs: Keyword arguments passed as CLI flags (--key=value).
    """
    check_uv()

    script_path = os.path.join(TOOLS_DIR, f"{script}.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    kwarg_flags = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", script_path] + list(args) + kwarg_flags

    print("Executing:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Tool execution failed with code {e.returncode}")
        sys.exit(1)


def list_tools():
    """
    List all available smolbox tools.
    """
    print("> Available tools:")
    for fname in os.listdir(TOOLS_DIR):
        if fname.endswith(".py") and not fname.startswith("_"):
            tool_name = fname.removesuffix(".py")
            print(f"  - {tool_name}")


def main():
    fire.Fire({
        "exec": exec_tool,
        "list": list_tools,
        })



if __name__ == "__main__":
    main()


