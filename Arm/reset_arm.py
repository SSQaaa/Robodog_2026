# -*- coding: utf-8 -*-
"""将舵机移动到配置中的复位姿态。"""

import argparse
import time

from config import load_config
from servo_driver import ServoBus


def parse_args():
    parser = argparse.ArgumentParser(description="Reset arm to params.json arm.reset_pose.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--open-gripper", action="store_true", help="Use gripper open value for servo 1.")
    parser.add_argument("--speed", type=int, default=None)
    parser.add_argument("--acc", type=int, default=None)
    parser.add_argument("--wait", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    arm_cfg = config["arm"]
    pose = {int(k): int(v) for k, v in arm_cfg.get("reset_pose", {}).items()}
    if args.open_gripper:
        pose[int(arm_cfg["ids"]["gripper"])] = int(arm_cfg["gripper"]["open"])

    # 先动底座和抓手旋转，再动机械臂关节，最后动抓手。
    order = [
        int(arm_cfg["ids"]["base"]),
        int(arm_cfg["ids"]["gripper_rotate"]),
        int(arm_cfg["ids"]["shoulder"]),
        int(arm_cfg["ids"]["elbow"]),
        int(arm_cfg["ids"]["wrist"]),
        int(arm_cfg["ids"]["gripper"]),
    ]

    bus = ServoBus(arm_cfg)
    try:
        print(f"[Reset] pose={pose}")
        for servo_id in order:
            if servo_id not in pose:
                continue
            bus.write_pos(
                servo_id,
                pose[servo_id],
                speed=args.speed if args.speed is not None else arm_cfg["moving_speed"],
                acc=args.acc if args.acc is not None else arm_cfg["moving_acc"],
            )
            time.sleep(0.08)
        time.sleep(float(args.wait))
        bus.print_status(order)
        print("[Reset] done")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
