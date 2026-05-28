# -*- coding: utf-8 -*-
"""使能或失能配置中的舵机扭矩。"""

import argparse

from config import all_servo_ids, load_config
from servo_driver import ServoBus


def parse_args():
    parser = argparse.ArgumentParser(description="Enable/disable ST3215 torque.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--enable", action="store_true", help="Enable torque instead of disabling.")
    parser.add_argument("--ids", nargs="*", type=int, help="Servo IDs. Defaults to all configured IDs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ids = args.ids if args.ids else all_servo_ids(config)
    bus = ServoBus(config["arm"])
    try:
        bus.set_torque_many(ids, args.enable)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
