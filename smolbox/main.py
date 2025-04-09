import os
import sys
import fire
import shutil
import subprocess
from smolbox.core.commons import next_state


TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def check_uv():
    """
    Check if the 'uv' CLI tool is installed and working.
    Exits the program if 'uv --version' fails.
    """
    try:
        _ = subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        print("Ensure uv is working properly! (uv --version) failed")
        sys.exit(1)


def exec_tool(script: str, *args, **kwargs):
    """
    Run a tool script from the tools directory using 'uv run'.

    Args:
        script (str): Script name without '.py' (e.g., 'param_tweak_sampler').
        args: Positional CLI arguments (e.g., subcommands like 'sample').
        kwargs: Keyword arguments passed as CLI flags (--key=value).
    """
    check_uv()

    script_path = os.path.join(TOOLS_DIR, f"{script}.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    kwarg_flags = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", "-q", script_path] + list(args) + kwarg_flags

    try:
        subprocess.run(cmd, check=True)
        next_state()
    except subprocess.CalledProcessError as e:
        print(f"Tool execution failed with code {e.returncode}")
        sys.exit(1)


def list_tools():
    """
    Print all available tools in the tools directory.
    """
    print("> Available tools:")
    for fname in os.listdir(TOOLS_DIR):
        if fname.endswith(".py") and not fname.startswith("_"):
            tool_name = fname.removesuffix(".py")
            print(f"  - {tool_name}")


def exec_experiments(script: str, *args, **kwargs):
    """
    Run an experiment script from the experiments directory using 'uv run'.

    Args:
        script (str): Script name without '.py' (e.g., 'ablation_study').
        args: Positional CLI arguments.
        kwargs: Keyword arguments passed as CLI flags (--key=value).
    """
    check_uv()

    script_path = os.path.join(EXP_DIR, f"{script}.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    kwarg_flags = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", "-q", script_path] + list(args) + kwarg_flags

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment execution failed with code {e.returncode}")
        sys.exit(1)


def list_experiments():
    """
    Print all available experiment scripts in the experiments directory.
    """
    print("> Available experiments:")
    for fname in os.listdir(EXP_DIR):
        if fname.endswith(".py") and not fname.startswith("_"):
            experiment_name = fname.removesuffix(".py")
            print(f"  - {experiment_name}")


def main():
    fire.Fire({
        "run": exec_tool,
        "ls": list_tools,
        "run-exp": exec_experiments,  # <- was calling exec_tool before, fixed
        "ls-exp": list_experiments,  # <- was calling list_tools before, fixed
    })


if __name__ == "__main__":
    main()
