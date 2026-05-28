# -*- coding: utf-8 -*-
import time

from arm_control import ArmControl, DEFAULT_CONFIG_PATH
from dog_control import (
    BACKWARD_SPEED,
    FORWARD_SPEED,
    PRE_GRASP_MOVE_SPEED_X,
    PRE_GRASP_MOVE_SPEED_Y,
    SIDE_SPEED,
    DogControl,
)
from vision_control import YoloDepthDetector

DASHBOARD_STATUS = {
    "A": "ABNORMAL",
    "B": "NORMAL",
    "C": "ABNORMAL",
    "D": "NORMAL",
}
CONFIG_PATH = DEFAULT_CONFIG_PATH
DRY_RUN = False

STATUS_TO_BLOCK = {
    "ABNORMAL": "Red",
    "NORMAL": "Green",
}

CENTER_TOLERANCE_BLOCK_PX = 100
PRE_GRASP_MOVE_SECONDS_X = 0.1
PRE_GRASP_MOVE_SECONDS_Y = 0.3
PRE_GRASP_MAX_ADJUST_SECONDS = 60.0
GRASP_R_LIMIT_MM = 430.0

FORWARD_SECONDS = 2.0
CENTER_TOLERANCE_BOX_PX = 30
MAX_ALIGN_SECONDS = 150.0
FINAL_BOX_FORWARD_SECONDS = 2.0

A_BACK_MOVE_SECONDS = 6.0

RETURN_FORWARD_SECONDS = 4.5

SIDE_MOVE_SECONDS = 1.7


class Task3Mission:
    def __init__(self, status_dict=None, config_path=CONFIG_PATH, dry_run=DRY_RUN):
        self.status_dict = status_dict or DASHBOARD_STATUS
        self.vision = YoloDepthDetector()
        self.arm = ArmControl(config_path=config_path, dry_run=dry_run)
        self.dog = DogControl()

    def start(self):
        self.vision.start()
        self.arm.start()
        self.dog.stand_up()
        return self

    def close(self):
        self.arm.close()
        self.vision.stop()
        self.dog.close()

    def run(self):
        for letter in ("A", "B", "C", "D"):
            if letter in self.status_dict:
                self.run_single_transfer(letter, self.status_dict[letter])

    # 单轮流程：抓取、转身、找箱子、放置、返回。
    def run_single_transfer(self, letter, status):
        block_class = self.decide_pick_what(status)
        print(f"[TASK3] {letter}={status}, pick {block_class}")

        block = self.adjust_before_grasp_1(block_class)
        time.sleep(0.5)
        block = self.adjust_before_grasp_2(block_class)
        # 充分时间冷却，防止机械臂的坐标计算是移动中的结果
        time.sleep(0.5)
        self.arm.pick_block(block_class, block, self.vision.color_intrinsics)
        time.sleep(0.5)

        # 转180
        self.dog.revolve_180()
        # 往前走
        self.dog.move(vx=FORWARD_SPEED, last_time=FORWARD_SECONDS, duration=0.3)
        time.sleep(0.5)

        self.approach_box_1(letter)
        time.sleep(0.5)
        self.approach_box_2(letter)
        time.sleep(0.5)
        self.arm.place_block()
        time.sleep(0.5)

        self.dog.revolve_180()
        time.sleep(0.5)
        if letter == 'A':
            self.dog.move(vy=-SIDE_SPEED, last_time=A_BACK_MOVE_SECONDS, duration=0.3)
        if letter == 'B':
            self.dog.move(vy=-SIDE_SPEED, last_time=A_BACK_MOVE_SECONDS-3.0, duration=0.3)

        self.dog.move(vx=FORWARD_SPEED, last_time=RETURN_FORWARD_SECONDS, duration=0.3)
        time.sleep(0.5)

    # 仪表盘状态决定抓红色还是绿色。
    def decide_pick_what(self, status):
        key = str(status).strip()
        if key not in STATUS_TO_BLOCK:
            raise ValueError(f"Unsupported dashboard status: {status}")
        return STATUS_TO_BLOCK[key]

    # 先让目标字母进入画面中心，再固定直走到箱子前。
    def adjust_before_grasp_1(self, block_class):
        print(f"[GraspAdjust_1] align to {block_class}")
        deadline = time.time() + PRE_GRASP_MAX_ADJUST_SECONDS
        last_seen = None

        while time.time() < deadline:
            frame, detections = self.vision.detect()
            if frame is None:
                continue

            matches = [det for det in detections if det.class_name == block_class]
            if not matches:
                self.dog.move(vx=BACKWARD_SPEED, last_time=0.2, duration=0.3)
                print(f"[GraspAdjust_1] {block_class} not found, move backward a little")
                time.sleep(0.1)
                continue

            _, frame_w = frame.shape[:2]
            matches.sort(key=lambda det: abs(det.center[0] - frame_w / 2))
            block = matches[0]
            last_seen = block

            error_x = block.center[0] - frame_w * 0.5
            print(f"[GraspAdjust_1] {block_class} error_x={error_x:.1f}px")

            if abs(error_x) > CENTER_TOLERANCE_BLOCK_PX:
                vy = PRE_GRASP_MOVE_SPEED_Y if error_x > 0 else -PRE_GRASP_MOVE_SPEED_Y
                self.dog.move(vy=vy, last_time=PRE_GRASP_MOVE_SECONDS_Y, duration=0.3)
                print(f"[GraspAdjust_1] move {'right' if vy > 0 else 'left'} to adjust")
                continue

            print("[GraspAdjust_1] target centered")
            return

        raise RuntimeError(f"Failed to reach block {block_class}; last_seen={last_seen}")
    
    # 前后校准，抓取前校准机器狗位置，让物块落在机械臂可抓范围内。
    def adjust_before_grasp_2(self, block_class):
        print(f"[GraspAdjust_2] start for {block_class}")
        deadline = time.time() + PRE_GRASP_MAX_ADJUST_SECONDS
        last_seen = None

        while time.time() < deadline:
            frame, detections = self.vision.detect()
            if frame is None:
                continue

            matches = [det for det in detections if det.class_name == block_class]
            if not matches:
                print(f"[GraspAdjust_2] {block_class} not found")
                time.sleep(0.1)
                continue
            
            _, frame_w = frame.shape[:2]
            matches.sort(key=lambda det: abs(det.center[0] - frame_w / 2))
            block = matches[0]
            last_seen = block

            if block.depth_mm is None:
                print("[GraspAdjust_2] depth invalid, move backward a little")
                self.dog.move(vx=-PRE_GRASP_MOVE_SPEED_X, last_time=PRE_GRASP_MOVE_SECONDS_X, duration=0.5)
                continue

            try:
                result = self.arm.compute_grasp_target(block, self.vision.color_intrinsics)
                r_mm = result["solution"].r_mm
            except Exception as exc:
                print(f"[GraspAdjust_2] arm target failed: {exc}; move forward a little")
                self.dog.move(vx=PRE_GRASP_MOVE_SPEED_X, last_time=PRE_GRASP_MOVE_SECONDS_X, duration=0.5)
                continue

            print(f"[GraspAdjust_2] r={r_mm:.1f}mm limit={GRASP_R_LIMIT_MM:.1f}mm")
            if r_mm > GRASP_R_LIMIT_MM:
                print("[GraspAdjust_2] target too far, move forward a little")
                self.dog.move(vx=PRE_GRASP_MOVE_SPEED_X, last_time=PRE_GRASP_MOVE_SECONDS_X, duration=0.5)
                continue

            print("[GraspAdjust_2] target ready")
            return block

        raise RuntimeError(f"Failed to adjust before grasp {block_class}; last_seen={last_seen}")

    # 先让目标字母进入画面中心，再固定直走到箱子前。
    def approach_box_1(self, letter):
        print(f"[Box] align to {letter}")
        deadline = time.time() + MAX_ALIGN_SECONDS
        last_seen = None

        while time.time() < deadline:
            frame, detections = self.vision.detect()
            if frame is None:
                continue

            matches = [det for det in detections if det.class_name == letter]
            if not matches:
                print(f"[Box] {letter} not found")
                time.sleep(0.1)
                continue

            matches.sort(key=lambda det: (det.area, det.conf), reverse=True)
            target = matches[0]
            last_seen = target

            _, frame_w = frame.shape[:2]
            error_x = target.center[0] - frame_w * 0.5
            print(f"[Box] {letter} error_x={error_x:.1f}px")

            if abs(error_x) > CENTER_TOLERANCE_BOX_PX:
                vy = SIDE_SPEED if error_x > 0 else -SIDE_SPEED
                self.dog.move(vy=vy, last_time=SIDE_MOVE_SECONDS, duration=0.3)
                continue

            print(f"[Box] {letter} centered")

            self.dog.stop()
            return

        raise RuntimeError(f"Failed to reach box {letter}; last_seen={last_seen}")
    
    # 先让目标字母进入画面中心，再固定直走到箱子前。
    def approach_box_2(self, letter):
        print(f"[Box] approach to {letter}")
        deadline = time.time() + MAX_ALIGN_SECONDS
        last_seen = None

        while time.time() < deadline:
            frame, detections = self.vision.detect()
            if frame is None:
                continue

            matches = [det for det in detections if det.class_name == letter]
            if not matches:
                print(f"[Box] {letter} not found")
                time.sleep(0.1)
                continue

            matches.sort(key=lambda det: (det.area, det.conf), reverse=True)
            target = matches[0]
            last_seen = target

            if target.depth_mm >= 700.0:
                self.dog.move(vx=FORWARD_SPEED, last_time=0.1, duration=0.3)
                print(f"[Box] {letter} depth={target.depth_mm:.1f}mm, move forward a little more")
                continue

            self.dog.stop()
            print(f"[Box] reached {letter}")
            return

        raise RuntimeError(f"Failed to reach box {letter}; last_seen={last_seen}")


def main():
    task3 = Task3Mission(status_dict=DASHBOARD_STATUS, config_path=CONFIG_PATH, dry_run=DRY_RUN)
    try:
        task3.start()
        task3.run()
    finally:
        task3.close()


if __name__ == "__main__":
    main()
