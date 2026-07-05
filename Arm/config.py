# -*- coding: utf-8 -*-
import json
import os
from copy import deepcopy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "params.json")


def load_config(path=None):
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path=None):
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def arm_ids(config):
    return deepcopy(config["arm"]["ids"])


def all_servo_ids(config):
    return sorted({int(value) for value in config["arm"]["ids"].values()})


def servo_cfg(config, servo_id):
    return config["arm"].get("servos", {}).get(str(int(servo_id)), {})


def clamp(value, low, high):
    return max(low, min(high, value))
