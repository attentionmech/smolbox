import os
import sys
import fire
import subprocess


TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")

def check_uv():
    try:
        _ = subprocess.run(["uv", "--version"], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        print("Ensure uv is working properly! (uv --version) failed")
        sys.exit(1)


def run(script: str, **kwargs):
    """
    Run a smolbox tool script using uv run.

    Args:
        script (str): Script name (e.g., param_tweak_sampler).
        kwargs: Arguments passed to the script as CLI flags.
    """
    check_uv()

    script_path = os.path.join(TOOLS_DIR, f"{script}.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    args = [f"--{k}={v}" for k, v in kwargs.items()]
    cmd = ["uv", "run", script_path] + args

    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        raise SystemExit(e.returncode)

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
        "run": run,
        "list": list_tools,
        })



if __name__ == "__main__":
    main()


