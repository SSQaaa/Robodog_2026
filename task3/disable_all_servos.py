# -*- coding: utf-8 -*-
"""Disable torque for all configured servos."""
"""
python disable_all_servos.py
python disable_all_servos.py --enable
失能指定id
python disable_all_servos.py --ids 3 4 5 6
"""
import argparse
import json
import os
import time

from scservo_sdk import PortHandler, sms_sts


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TORQUE_ENABLE_ADDR = 40


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def configured_servo_ids(config):
    ids = config.get("arm", {}).get("ids", {})
    return sorted({int(value) for value in ids.values()})


def write_torque(packet, servo_id, enabled):
    value = 1 if enabled else 0
    result, error = packet.write1ByteTxRx(int(servo_id), TORQUE_ENABLE_ADDR, value)
    return result, error


def main():
    parser = argparse.ArgumentParser(description="Disable or enable all configured servos.")
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "calibration.json"),
        help="Path to calibration.json.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable torque instead of disabling it.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        help="Optional explicit servo IDs. Defaults to arm.ids in calibration.json.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    arm_cfg = config["arm"]
    servo_ids = args.ids if args.ids else configured_servo_ids(config)
    if not servo_ids:
        raise RuntimeError("No servo IDs found. Check arm.ids in calibration.json or pass --ids.")

    port = PortHandler(arm_cfg["devicename"])
    packet = sms_sts(port)

    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {arm_cfg['devicename']}")
    if not port.setBaudRate(int(arm_cfg["baudrate"])):
        port.closePort()
        raise RuntimeError(f"Failed to set baudrate: {arm_cfg['baudrate']}")

    action = "enable" if args.enable else "disable"
    try:
        print(f"[Servo] {action} torque for IDs: {servo_ids}")
        for servo_id in servo_ids:
            result, error = write_torque(packet, servo_id, args.enable)
            print(f"[Servo] id={servo_id}, result={result}, error={error}")
            time.sleep(0.05)
    finally:
        port.closePort()
        print("[Servo] port closed")


if __name__ == "__main__":
    main()
