# -*- coding: utf-8 -*-
"""Move the arm back to the vendor default reset pose."""

import argparse
import os
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts  # noqa: E402


DEFAULT_POSE = {
    1: 2137,  # gripper closed
    2: 2043,  # gripper rotation center
    3: 2589,  # link 3 init
    4: 1886,   # link 4 init
    5: 2100,  # link 5 init
    6: 2047,  # base rotation center
}


def write_servo(packet, servo_id, position, speed, acc):
    result, error = packet.WritePosEx(int(servo_id), int(position), int(speed), int(acc))
    if result != COMM_SUCCESS:
        print(packet.getTxRxResult(result))
    if error != 0:
        print(packet.getRxPacketError(error))
    print(f"[Reset] ID {servo_id} -> {position}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reset SCServo robot arm pose")
    parser.add_argument("--dev", default="/dev/ttyUSB0", help="Servo serial device")
    parser.add_argument("--baudrate", type=int, default=500000)
    parser.add_argument("--speed", type=int, default=600)
    parser.add_argument("--acc", type=int, default=30)
    parser.add_argument("--open-gripper", action="store_true", help="Use gripper open value instead of closed")
    parser.add_argument("--wait", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    pose = dict(DEFAULT_POSE)
    if args.open_gripper:
        pose[1] = 2047

    port = PortHandler(args.dev)
    packet = sms_sts(port)

    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {args.dev}")
    if not port.setBaudRate(args.baudrate):
        raise RuntimeError(f"Failed to set baudrate: {args.baudrate}")

    try:
        # Move base and wrist first, then arm joints, then gripper.
        for servo_id in (6, 2, 5, 4, 3, 1):
            write_servo(packet, servo_id, pose[servo_id], args.speed, args.acc)
            time.sleep(0.08)
        time.sleep(args.wait)
        print("[Reset] done")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
