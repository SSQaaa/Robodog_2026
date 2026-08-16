# -*- coding: utf-8 -*-
"""持续打印舵机位置和电流。"""

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
        return "WARN"
    return "OK"


def fit_terminal_line(line):
    width = shutil.get_terminal_size((120, 20)).columns
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
            else:
                print(line, flush=True)
            time.sleep(max(0.02, float(args.interval)))
    except KeyboardInterrupt:
        if args.single_line:
            print()
        print("[Monitor] stopped")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
