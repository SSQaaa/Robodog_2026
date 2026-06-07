# -*- coding: utf-8 -*-
"""Restore basic SCServo/ST3215 parameters for one servo."""

import argparse
import time

from config import DEFAULT_CONFIG_PATH, load_config
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import (
    SMS_STS_LOCK,
    SMS_STS_MAX_ANGLE_LIMIT_L,
    SMS_STS_MIN_ANGLE_LIMIT_L,
    SMS_STS_MODE,
    SMS_STS_OFS_L,
    SMS_STS_TORQUE_ENABLE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Restore basic servo parameters.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to params.json.")
    parser.add_argument("--id", type=int, default=5, help="Servo ID to restore.")
    parser.add_argument("--min", type=int, default=0, help="Internal min angle limit.")
    parser.add_argument("--max", type=int, default=4095, help="Internal max angle limit.")
    parser.add_argument("--mode", type=int, default=0, help="0=position mode, 1=wheel/speed mode.")
    parser.add_argument("--offset", type=int, default=None, help="Optional offset to write. Omitted means keep current offset.")
    parser.add_argument("--yes", action="store_true", help="Actually write parameters.")
    return parser.parse_args()


def error_text(error):
    names = []
    if error & 1:
        names.append("voltage")
    if error & 2:
        names.append("angle")
    if error & 4:
        names.append("overheat")
    if error & 8:
        names.append("over-current")
    if error & 32:
        names.append("overload")
    return "|".join(names) if names else "none"


def check_comm(label, result, error):
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{label} failed: result={result}, error={error}")
    if error:
        print(f"[Warn] {label} servo_error={error} ({error_text(error)})")


def read1(packet, servo_id, addr):
    value, result, error = packet.read1ByteTxRx(servo_id, addr)
    check_comm(f"read1 addr={addr}", result, error)
    return int(value)


def read2(packet, servo_id, addr, signed=False):
    value, result, error = packet.read2ByteTxRx(servo_id, addr)
    check_comm(f"read2 addr={addr}", result, error)
    if signed:
        return int(packet.scs_tohost(value, 15))
    return int(value)


def write1(packet, servo_id, addr, value):
    result, error = packet.write1ByteTxRx(servo_id, addr, int(value))
    check_comm(f"write1 addr={addr} value={value}", result, error)


def write2(packet, servo_id, addr, value):
    result, error = packet.write2ByteTxRx(servo_id, addr, int(value))
    check_comm(f"write2 addr={addr} value={value}", result, error)


def print_params(packet, servo_id, label):
    print(f"[{label}]")
    print(f"  torque={read1(packet, servo_id, SMS_STS_TORQUE_ENABLE)}")
    print(f"  eprom_lock={read1(packet, servo_id, SMS_STS_LOCK)}")
    print(f"  mode={read1(packet, servo_id, SMS_STS_MODE)}")
    print(f"  internal_min={read2(packet, servo_id, SMS_STS_MIN_ANGLE_LIMIT_L)}")
    print(f"  internal_max={read2(packet, servo_id, SMS_STS_MAX_ANGLE_LIMIT_L)}")
    print(f"  offset={read2(packet, servo_id, SMS_STS_OFS_L, signed=True)}")


def main():
    args = parse_args()
    config = load_config(args.config)
    arm_cfg = config["arm"]
    servo_id = int(args.id)

    port = PortHandler(arm_cfg["devicename"])
    packet = sms_sts(port)
    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {arm_cfg['devicename']}")
    if not port.setBaudRate(int(arm_cfg["baudrate"])):
        port.closePort()
        raise RuntimeError(f"Failed to set servo baudrate: {arm_cfg['baudrate']}")

    try:
        print(f"[Restore] opened {arm_cfg['devicename']} @ {arm_cfg['baudrate']}")
        print(f"[Restore] id={servo_id}")
        print_params(packet, servo_id, "Before")

        if not args.yes:
            print("[Restore] dry run only. Add --yes to write.")
            return

        print("[Restore] torque off")
        write1(packet, servo_id, SMS_STS_TORQUE_ENABLE, 0)
        time.sleep(0.1)

        print("[Restore] unlock EPROM")
        packet.unLockEprom(servo_id)
        time.sleep(0.1)

        print(f"[Restore] write mode={args.mode}")
        write1(packet, servo_id, SMS_STS_MODE, args.mode)
        print(f"[Restore] write internal min/max={args.min}/{args.max}")
        write2(packet, servo_id, SMS_STS_MIN_ANGLE_LIMIT_L, args.min)
        write2(packet, servo_id, SMS_STS_MAX_ANGLE_LIMIT_L, args.max)
        if args.offset is not None:
            print(f"[Restore] write offset={args.offset}")
            write2(packet, servo_id, SMS_STS_OFS_L, packet.scs_toscs(int(args.offset), 15))

        time.sleep(0.1)
        print("[Restore] lock EPROM")
        packet.LockEprom(servo_id)
        time.sleep(0.1)

        print_params(packet, servo_id, "After")
        print("[Restore] done. Power-cycle the servo before testing motion.")
    finally:
        port.closePort()
        print("[Restore] port closed")


if __name__ == "__main__":
    main()
