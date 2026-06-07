# -*- coding: utf-8 -*-
"""Probe the goal-position register without necessarily moving the servo."""

import argparse
import time

from config import DEFAULT_CONFIG_PATH, load_config
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import (
    SMS_STS_ACC,
    SMS_STS_GOAL_POSITION_L,
    SMS_STS_GOAL_SPEED_L,
    SMS_STS_PRESENT_CURRENT_L,
    SMS_STS_PRESENT_POSITION_L,
    SMS_STS_TORQUE_ENABLE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Write/read a servo goal position for diagnosis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to params.json.")
    parser.add_argument("--id", type=int, default=5, help="Servo ID to probe.")
    parser.add_argument("--target", type=int, default=3375, help="Goal position to write.")
    parser.add_argument("--speed", type=int, default=100, help="Goal speed to write.")
    parser.add_argument("--acc", type=int, default=5, help="Goal acceleration to write.")
    parser.add_argument("--write", action="store_true", help="Actually write the target register.")
    parser.add_argument("--torque-off", action="store_true", help="Disable torque before writing target.")
    parser.add_argument("--yes", action="store_true", help="Confirm writing the target register.")
    parser.add_argument("--watch", type=float, default=0.0, help="Seconds to keep reading after write/read.")
    parser.add_argument("--interval", type=float, default=0.1, help="Watch sample interval in seconds.")
    parser.add_argument("--current-limit", type=int, default=120, help="Torque off if abs(current) reaches this value.")
    return parser.parse_args()


def check(label, result, error):
    if result != COMM_SUCCESS or error:
        raise RuntimeError(f"{label} failed: result={result}, error={error}")


def check_comm(label, result, error):
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{label} failed: result={result}, error={error}")
    if error:
        print(f"[Warn] {label} servo_error={error} ({error_text(error)})")


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


def read1(packet, servo_id, addr):
    value, result, error = packet.read1ByteTxRx(servo_id, addr)
    if result != COMM_SUCCESS:
        raise RuntimeError(f"read1 addr={addr} failed: result={result}, error={error}")
    if error:
        print(f"[Warn] read1 addr={addr} servo_error={error} ({error_text(error)})")
    return int(value)


def read2(packet, servo_id, addr, signed=False):
    value, result, error = packet.read2ByteTxRx(servo_id, addr)
    if result != COMM_SUCCESS:
        raise RuntimeError(f"read2 addr={addr} failed: result={result}, error={error}")
    if error:
        print(f"[Warn] read2 addr={addr} servo_error={error} ({error_text(error)})")
    if signed:
        return int(packet.scs_tohost(value, 15))
    return int(value)


def write_target(packet, servo_id, target, speed, acc):
    txpacket = [
        int(acc),
        packet.scs_lobyte(int(target)),
        packet.scs_hibyte(int(target)),
        0,
        0,
        packet.scs_lobyte(int(speed)),
        packet.scs_hibyte(int(speed)),
    ]
    result, error = packet.writeTxRx(int(servo_id), SMS_STS_ACC, len(txpacket), txpacket)
    check("write target", result, error)


def print_snapshot(packet, servo_id, label):
    goal = read2(packet, servo_id, SMS_STS_GOAL_POSITION_L)
    speed = read2(packet, servo_id, SMS_STS_GOAL_SPEED_L)
    present = read2(packet, servo_id, SMS_STS_PRESENT_POSITION_L, signed=True)
    current = read2(packet, servo_id, SMS_STS_PRESENT_CURRENT_L, signed=True)
    torque = read1(packet, servo_id, SMS_STS_TORQUE_ENABLE)
    print(
        f"[{label}] torque={torque} goal_pos={goal} "
        f"goal_speed={speed} present_pos={present} present_current={current}"
    )
    return {
        "torque": torque,
        "goal_pos": goal,
        "goal_speed": speed,
        "present_pos": present,
        "present_current": current,
    }


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
        print(f"[Probe] opened {arm_cfg['devicename']} @ {arm_cfg['baudrate']}")
        print_snapshot(packet, servo_id, "Before")

        if args.write:
            if not args.yes:
                raise RuntimeError("Refusing to write without --yes")
            if args.torque_off:
                result, error = packet.write1ByteTxRx(servo_id, SMS_STS_TORQUE_ENABLE, 0)
                check_comm("torque off", result, error)
                time.sleep(0.1)
                print_snapshot(packet, servo_id, "After torque off")

            print(
                f"[Probe] write id={servo_id} target={args.target} "
                f"speed={args.speed} acc={args.acc}"
            )
            write_target(packet, servo_id, args.target, args.speed, args.acc)
            time.sleep(0.1)
            print_snapshot(packet, servo_id, "After write")

        if args.watch > 0:
            deadline = time.time() + float(args.watch)
            sample = 0
            while time.time() < deadline:
                sample += 1
                time.sleep(max(0.02, float(args.interval)))
                snapshot = print_snapshot(packet, servo_id, f"Watch {sample}")
                current = snapshot["present_current"]
                if abs(int(current)) >= int(args.current_limit):
                    print(f"[Probe] current limit reached: {current}; torque off")
                    result, error = packet.write1ByteTxRx(servo_id, SMS_STS_TORQUE_ENABLE, 0)
                    check_comm("torque off after current limit", result, error)
                    print_snapshot(packet, servo_id, "After safety torque off")
                    break
    finally:
        port.closePort()
        print("[Probe] port closed")


if __name__ == "__main__":
    main()
