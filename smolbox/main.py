import os
import subprocess
import sys

import fire
from smolbox.core.state_manager import (
    next_state,
    reset_state,
    set,
    get,
    init_state,
    print_state,
    list_models,
)

from smolbox.core.tool_manager import exec_tool, INTERNAL_COMMANDS
from smolbox.config import DEFAULT_TOOLS_DIR




def main():
    if len(sys.argv) < 2:
        print(
            "Usage: smolbox <toolname> [args]  — or —  smolbox "
            + " | ".join(INTERNAL_COMMANDS.keys())
        )
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
