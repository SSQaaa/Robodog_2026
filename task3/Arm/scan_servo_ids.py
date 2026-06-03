# -*- coding: utf-8 -*-
"""Scan SCServo/ST3215 IDs on the configured serial bus."""

import argparse
import time

from config import DEFAULT_CONFIG_PATH, load_config
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts


DEFAULT_BAUDRATES = [500000, 1000000, 115200, 250000]


def parse_ids(text):
    ids = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    return sorted(i for i in ids if 0 <= i <= 252)


def scan_once(port_name, baudrate, ids, delay_s):
    port = PortHandler(port_name)
    packet = sms_sts(port)
    found = []

    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {port_name}")
    try:
        if not port.setBaudRate(int(baudrate)):
            raise RuntimeError(f"Failed to set baudrate: {baudrate}")

        print(f"[Scan] port={port_name} baudrate={baudrate} ids={ids[0]}..{ids[-1]}")
        for servo_id in ids:
            pos, result, error = packet.ReadPos(int(servo_id))
            if result == COMM_SUCCESS and not error:
                found.append((servo_id, int(pos)))
                print(f"[FOUND] id={servo_id} pos={int(pos)}")
            if delay_s > 0:
                time.sleep(delay_s)
    finally:
        port.closePort()

    return found


def main():
    parser = argparse.ArgumentParser(description="Scan SCServo/ST3215 servo IDs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to params.json.")
    parser.add_argument("--port", default=None, help="Serial port. Defaults to arm.devicename in params.json.")
    parser.add_argument("--baudrate", type=int, default=None, help="Baudrate. Defaults to arm.baudrate in params.json.")
    parser.add_argument(
        "--try-common-baudrates",
        action="store_true",
        help="Try 500000, 1000000, 115200, and 250000 instead of only one baudrate.",
    )
    parser.add_argument("--ids", default="0-20", help="IDs to scan, e.g. 0-20, 1-6, or 1,2,5.")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between ID probes in seconds.")
    args = parser.parse_args()

    config = load_config(args.config)
    arm_cfg = config["arm"]
    port_name = args.port or arm_cfg["devicename"]
    ids = parse_ids(args.ids)
    if not ids:
        raise ValueError("No valid IDs to scan.")

    baudrates = DEFAULT_BAUDRATES if args.try_common_baudrates else [args.baudrate or int(arm_cfg["baudrate"])]
    total_found = []
    for baudrate in baudrates:
        found = scan_once(port_name, baudrate, ids, args.delay)
        total_found.extend((baudrate, servo_id, pos) for servo_id, pos in found)

    if not total_found:
        print("[Scan] no servo replied")
        print("[Hint] Check power, wiring, port name, baudrate, and whether the servo ID is outside the scanned range.")
        return

    print("[Scan] summary:")
    for baudrate, servo_id, pos in total_found:
        print(f"  baudrate={baudrate} id={servo_id} pos={pos}")


if __name__ == "__main__":
    main()
