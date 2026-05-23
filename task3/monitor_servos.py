# -*- coding: utf-8 -*-
<<<<<<< HEAD
"""Continuously print servo current feedback and position."""

import argparse
import json
import os
import shutil
import time

from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import SMS_STS_PRESENT_CURRENT_L


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_MA_PER_UNIT = 6.5


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def configured_servo_ids(config):
    ids = config.get("arm", {}).get("ids", {})
    return sorted({int(value) for value in ids.values()})


def current_level(current_units):
    if current_units >= 300:
        return "DANGER"
    if current_units >= 250:
        return "HIGH"
    if current_units >= 200:
=======
"""Continuously print servo position and current."""

import argparse
import shutil
import time

from config import all_servo_ids, load_config
from servo_driver import ServoBus


def current_level(current_units, warn=50, high=100, danger=150):
    if current_units is None:
        return "UNKNOWN"
    value = abs(int(current_units))
    if value >= int(danger):
        return "DANGER"
    if value >= int(high):
        return "HIGH"
    if value >= int(warn):
>>>>>>> replace task3 with new grasp pipeline
        return "WARN"
    return "OK"


def fit_terminal_line(line):
    width = shutil.get_terminal_size((120, 20)).columns
<<<<<<< HEAD
    if width <= 1:
        return line
    max_len = width - 1
    if len(line) <= max_len:
        return line
    if max_len <= 3:
        return line[:max_len]
    return line[: max_len - 3] + "..."


def read_status(packet, servo_id):
    position, pos_result, pos_error = packet.ReadPos(servo_id)
    if pos_result != COMM_SUCCESS or pos_error != 0:
        return None, None, f"pos_result={pos_result}, pos_error={pos_error}"

    current_raw, cur_result, cur_error = packet.read2ByteTxRx(servo_id, SMS_STS_PRESENT_CURRENT_L)
    if cur_result != COMM_SUCCESS or cur_error != 0:
        return None, None, f"cur_result={cur_result}, cur_error={cur_error}"

    current_units = packet.scs_tohost(current_raw, 15)
    return current_units, position, None


def main():
    parser = argparse.ArgumentParser(description="Monitor servo current and position.")
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "calibration.json"),
        help="Path to calibration.json.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        help="Optional explicit servo IDs. Defaults to arm.ids in calibration.json.",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="Print interval in seconds.")
    parser.add_argument(
        "--single-line",
        "--overwrite",
        action="store_true",
        help="Overwrite one terminal line instead of printing a new line each sample.",
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

    print(f"[Monitor] ids={servo_ids}, interval={args.interval:.2f}s")
    print("[Monitor] Press Ctrl+C to stop.")

    last_line_len = 0
    try:
        while True:
            stamp = time.strftime("%H:%M:%S")
            parts = []
            for servo_id in servo_ids:
                try:
                    current_units, position, error = read_status(packet, servo_id)
                    if error is not None:
                        parts.append(f"id{servo_id}: ERR {error}")
                        continue
                    current_ma = float(current_units) * CURRENT_MA_PER_UNIT
                    parts.append(
                        "id{}: cur={}({:.1f}mA,{}) pos={}".format(
                            servo_id,
                            current_units,
                            current_ma,
                            current_level(current_units),
                            position,
                        )
                    )
                except Exception as exc:
                    parts.append(f"id{servo_id}: ERR {exc}")
            line = f"[{stamp}] " + " | ".join(parts)
            if args.single_line:
                line = fit_terminal_line(line)
                padding = " " * max(0, last_line_len - len(line))
                print("\r\033[K" + line + padding, end="", flush=True)
                last_line_len = len(line)
=======
    if len(line) < width:
        return line
    return line[: max(1, width - 4)] + "..."


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor ST3215 position/current.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ids", nargs="*", type=int, help="Servo IDs. Defaults to all configured IDs.")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--single-line", "--overwrite", action="store_true")
    parser.add_argument("--warn-current", type=int, default=80, help="Current units for WARN.")
    parser.add_argument("--high-current", type=int, default=150, help="Current units for HIGH.")
    parser.add_argument("--danger-current", type=int, default=250, help="Current units for DANGER.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ids = args.ids if args.ids else all_servo_ids(config)
    bus = ServoBus(config["arm"])
    last_len = 0
    try:
        while True:
            parts = []
            for servo_id in ids:
                status = bus.read_status(servo_id)
                if status.error:
                    parts.append(f"id{servo_id}:ERR {status.error}")
                    continue
                level = current_level(
                    status.current_units,
                    warn=args.warn_current,
                    high=args.high_current,
                    danger=args.danger_current,
                )
                current_ma = None
                if status.current_units is not None:
                    current_ma = status.current_units * bus.current_ma_per_unit
                current_text = "n/a" if current_ma is None else f"{current_ma:.0f}mA"
                parts.append(
                    "id{}:pos={} cur={}({}) state={} tmp={}".format(
                        servo_id,
                        status.position,
                        status.current_units,
                        current_text,
                        level,
                        status.temperature_c,
                    )
                )
            line = time.strftime("[%H:%M:%S] ") + " | ".join(parts)
            if args.single_line:
                line = fit_terminal_line(line)
                print("\r\033[K" + line + " " * max(0, last_len - len(line)), end="", flush=True)
                last_len = len(line)
>>>>>>> replace task3 with new grasp pipeline
            else:
                print(line, flush=True)
            time.sleep(max(0.02, float(args.interval)))
    except KeyboardInterrupt:
        if args.single_line:
            print()
<<<<<<< HEAD
        print("\n[Monitor] stopped")
    finally:
        port.closePort()
        print("[Monitor] port closed")
=======
        print("[Monitor] stopped")
    finally:
        bus.close()
>>>>>>> replace task3 with new grasp pipeline


if __name__ == "__main__":
    main()
