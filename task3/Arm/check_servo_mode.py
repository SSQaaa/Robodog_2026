# -*- coding: utf-8 -*-
"""Read SCServo/ST3215 mode register for one or more servos."""

import argparse

from config import DEFAULT_CONFIG_PATH, all_servo_ids, load_config
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import SMS_STS_MODE


def parse_args():
    parser = argparse.ArgumentParser(description="Read servo mode register address 33.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to params.json.")
    parser.add_argument("--ids", type=int, nargs="*", help="Servo IDs to check. Defaults to all configured IDs.")
    return parser.parse_args()


def describe_mode(value):
    if value == 0:
        return "position mode"
    if value == 1:
        return "wheel/speed mode"
    return "unknown/custom mode"


def main():
    args = parse_args()
    config = load_config(args.config)
    arm_cfg = config["arm"]
    ids = args.ids if args.ids else all_servo_ids(config)

    port = PortHandler(arm_cfg["devicename"])
    packet = sms_sts(port)
    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {arm_cfg['devicename']}")
    if not port.setBaudRate(int(arm_cfg["baudrate"])):
        port.closePort()
        raise RuntimeError(f"Failed to set servo baudrate: {arm_cfg['baudrate']}")

    try:
        print(f"[ServoMode] opened {arm_cfg['devicename']} @ {arm_cfg['baudrate']}")
        for servo_id in ids:
            mode, result, error = packet.read1ByteTxRx(int(servo_id), SMS_STS_MODE)
            if result != COMM_SUCCESS or error:
                print(f"id{servo_id}: ERR result={result} error={error}")
                continue
            print(f"id{servo_id}: mode={mode} ({describe_mode(int(mode))})")
    finally:
        port.closePort()
        print("[ServoMode] port closed")


if __name__ == "__main__":
    main()
