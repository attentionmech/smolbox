import os
import json
from uuid import uuid4
from datetime import datetime

AUTORESOLVE = "<AUTO>"

ALLOWED_KEYS = ["model_path", "model_name", "output_model_path", "output_model_name"]
WRITABLE_KEYS = ["output_model_path"]

SMOLBOX_DIR = os.path.join(os.getcwd(), ".smolbox")
STATE_FILE = os.path.join(SMOLBOX_DIR, "state.json")
STATE_HISTORY_FILE = os.path.join(SMOLBOX_DIR, "state_history.jsonl")

def now():
    return datetime.utcnow().isoformat() + "Z"  # ISO 8601 in UTC

def ensure_smolbox_dir():
    if not os.path.exists(SMOLBOX_DIR):
        os.makedirs(SMOLBOX_DIR)

def resolve(key_name, key_value, write=False):
    
    if key_value != AUTORESOLVE:
        if key_value:
            dikt = get_current_state()
            dikt[key_name] = key_value
            save_current_state(dikt)
        return key_value
    
    if key_name not in ALLOWED_KEYS:
        print(f"Invalid key name: {key_name} for state.")
        raise ValueError("Invalid key name for state. Allowed keys are: " + ", ".join(ALLOWED_KEYS))

    dikt = get_current_state()

    if not write:
        if key_name in dikt and dikt[key_name] is not None:
            return dikt[key_name]
        else:
            raise ValueError(f"Could not resolve {key_name}")
    else:
        if key_name in WRITABLE_KEYS:
            if key_name in dikt and dikt[key_name] is not None:
                return dikt[key_name]                # raise ValueError(f"{key_name} already resolved to {dikt[key_name]}")
            else:
                new_folder_name = str(uuid4())
                new_folder_path = os.path.join(SMOLBOX_DIR, new_folder_name)
                os.makedirs(new_folder_path)
                dikt[key_name] = new_folder_path

                if "created_at" not in dikt:
                    dikt["created_at"] = now()
                dikt["updated_at"] = now()

                save_current_state(dikt)
                return new_folder_path

def get_current_state():
    ensure_smolbox_dir()
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump({}, f)
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_current_state(state_dict):
    ensure_smolbox_dir()
    state_dict["updated_at"] = now()
    if "created_at" not in state_dict:
        state_dict["created_at"] = now()

    with open(STATE_FILE, "w") as f:
        json.dump(state_dict, f, indent=2)

    with open(STATE_HISTORY_FILE, "a") as f:
        f.write(json.dumps(state_dict) + "\n")

def next_state():
    current_state = get_current_state()
    save_current_state(current_state)  # log before mutating

    current_state["model_path"] = current_state.get("output_model_path")
    current_state["model_name"] = current_state.get("output_model_name")
    current_state["output_model_path"] = None
    current_state["output_model_name"] = None

    current_state["updated_at"] = now()

    save_current_state(current_state)
    return current_state
