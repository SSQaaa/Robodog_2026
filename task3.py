# -*- coding: utf-8 -*-
import time

from arm_control import ArmControl, DEFAULT_CONFIG_PATH
from project_config import (
    BACKWARD_SPEED,
    DEFAULT_DASHBOARD_STATUS,
    FORWARD_SPEED,
    PRE_GRASP_MOVE_SPEED_X,
    PRE_GRASP_MOVE_SPEED_Y,
    SIDE_SPEED,
)
from tools.motion import DogControl
from tools.vision import YoloDepthDetector

DASHBOARD_STATUS = DEFAULT_DASHBOARD_STATUS

STATUS_TO_BLOCK = {
    "ABNORMAL": "Red",
    "NORMAL": "Green",
}

CENTER_TOLERANCE_BLOCK_PX = 100  # 中心值+-100就认为绿/红色物块在画面中心，主要是因为离得比较近
PRE_GRASP_MOVE_SECONDS_X = 0.1   # 抓物块时候狗往前/后移动的时间，调整物块距离
PRE_GRASP_MOVE_SECONDS_Y = 0.3   # 抓物块时候狗往左/右移动的时间，调整物块在画面中的位置
PRE_GRASP_MAX_ADJUST_SECONDS = 30.0  # 抓取前最大调整时间，超过这个时间就放弃抓取，重新开始流程。因为如果调整时间过长可能是识别有问题或者位置太偏了，继续调整可能也很难成功。
GRASP_R_LIMIT_MM = 395.0
GRASP2_MISSING_RECENTER_COUNT = 5

FORWARD_SECONDS = 2.5
CENTER_TOLERANCE_BOX_PX = 30  # 中心值+-30px就认为ABCD在画面中心
MAX_ALIGN_SECONDS = 60.0
BOX_LOST_NEAR_DEPTH_MM = 300.0
BOX_SEENED_DEPTH_MM = 1000.0

BACK_OFFSET_TIME = 4.0

RETURN_FORWARD_SECONDS = 4.5

SIDE_MOVE_SECONDS = 1.0


class Task3:
    def __init__(self, status_dict=None, dog=None, config_path=DEFAULT_CONFIG_PATH, dry_run=False):
        self.status_dict = dict(status_dict or DASHBOARD_STATUS)
        self.vision = YoloDepthDetector()
        self.arm = ArmControl(config_path=config_path, dry_run=dry_run)
        self.dog = dog
        self._own_dog = dog is None

    def start(self):
        self.vision.start()
        self.arm.start()
        if self.dog is None:
            self.dog = DogControl()
            self.dog.stand_up()
        return self

    def close(self):
        self.arm.close()
        self.vision.stop()
        if self._own_dog and self.dog is not None:
            self.dog.close()
            self.dog = None

    def run(self):
        abnormal_letters = [
            letter
            for letter in ("A", "B", "C", "D")
            if self.status_dict.get(letter) == "ABNORMAL"
        ]
        if not abnormal_letters:
            print("[TASK3] no abnormal red block to pick")
            return

        for index, letter in enumerate(abnormal_letters):
            self.run_single_transfer(
                letter,
                self.status_dict[letter],
                should_return=index < len(abnormal_letters) - 1,
            )

    # 单轮流程：抓取、转身、找箱子、放置、返回。
    def run_single_transfer(self, letter, status, should_return=True):
        block_class = self.decide_pick_what(status)
        print(f"[TASK3] {letter}={status}, pick {block_class}")

        # 抓方块
        self.grasp(block_class)
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

        if not should_return:
            return

        self.dog.revolve_180()
        time.sleep(0.5)
        if letter == 'A':
            self.dog.move(vy=-SIDE_SPEED, last_time=BACK_OFFSET_TIME, duration=0.3)
        if letter == 'B':
            self.dog.move(vy=-SIDE_SPEED, last_time=BACK_OFFSET_TIME - 3.0, duration=0.3)

        self.dog.move(vx=FORWARD_SPEED, last_time=RETURN_FORWARD_SECONDS, duration=0.3)
        time.sleep(0.5)

    # 仪表盘状态决定抓红色还是绿色。
    def decide_pick_what(self, status):
        key = str(status).strip()
        if key not in STATUS_TO_BLOCK:
            raise ValueError(f"Unsupported dashboard status: {status}")
        return STATUS_TO_BLOCK[key]

    def detect_matches(self, class_name):
        frame, detections = self.vision.detect()
        if frame is None:
            return None, []
        return frame, [det for det in detections if det.class_name == class_name]

    def refind_target(self, class_name, back_time=0.3):
        self.dog.stop()
        time.sleep(0.5)
        frame, matches = self.detect_matches(class_name)
        if matches:
            print(f"[Recover] {class_name} found after stand still")
            return frame, matches
        print(f"[Recover] {class_name} not found, move backward")
        self.dog.move(vx=BACKWARD_SPEED, last_time=back_time, duration=0.3)
        time.sleep(0.5)
        return frame, []

    def x_move_by_depth(self, depth_mm):
        if depth_mm is not None and float(depth_mm) < 500.0:
            return 7000, 0.1
        return 10000, 0.3
    
    # 抓取物块，抓取失败时重新校准并重新计算视觉坐标。
    def grasp(self, block_class):
        attempt = 1
        while True:
            print(f"[PickRetry] attempt {attempt}")
            try:
                self.adjust_before_grasp_1(block_class)
                time.sleep(0.5)
                block = self.adjust_before_grasp_2(block_class)
                picked = self.arm.pick_block(block_class, block, self.vision.color_intrinsics)
            except Exception as exc:
                picked = False
                print(f"[PickRetry] adjust/pick failed with exception: {exc}")
            if picked:
                return
            print("[PickRetry] grasp failed, adjust and try again")
            attempt += 1
            time.sleep(0.5)
        
    # 先让目标字母进入画面中心，再固定直走到箱子前。
    def adjust_before_grasp_1(self, block_class):
        print(f"[GraspAdjust_1] align to {block_class}")
        deadline = time.time() + PRE_GRASP_MAX_ADJUST_SECONDS
        last_seen = None

        while time.time() < deadline:
            frame, matches = self.detect_matches(block_class)
            if frame is None:
                continue

            if not matches:
                frame, matches = self.refind_target(block_class, back_time=0.2)
                if frame is None or not matches:
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
        missing_count = 0

        while time.time() < deadline:
            frame, matches = self.detect_matches(block_class)
            if frame is None:
                continue

            if not matches:
                missing_count += 1
                print(f"[GraspAdjust_2] {block_class} not found")
                if missing_count >= GRASP2_MISSING_RECENTER_COUNT:
                    frame, matches = self.refind_target(block_class, back_time=0.2)
                    if frame is None or not matches:
                        raise RuntimeError(f"{block_class} lost during grasp adjust 2; rerun coarse alignment")
                    missing_count = 0
                time.sleep(0.1)
                if not matches:
                    continue
            
            missing_count = 0
            _, frame_w = frame.shape[:2]
            matches.sort(key=lambda det: abs(det.center[0] - frame_w / 2))
            block = matches[0]
            last_seen = block

            error_x = block.center[0] - frame_w * 0.5
            print(f"[GraspAdjust_2] {block_class} error_x={error_x:.1f}px")

            if abs(error_x) > CENTER_TOLERANCE_BLOCK_PX:
                    raise RuntimeError(
                        f"{block_class} lateral error too large in grasp adjust 2: "
                        f"error_x={error_x:.1f}px; rerun coarse alignment"
                    )

            if block.depth_mm is None:
                print("[GraspAdjust_2] depth invalid, move backward a little")
                self.dog.move(vx=-PRE_GRASP_MOVE_SPEED_X, last_time=PRE_GRASP_MOVE_SECONDS_X, duration=0.5)
                continue

            try:
                plan = self.arm.compute_pick_plan(block, self.vision.color_intrinsics)
                r_mm = plan["target"]["solution"].r_mm
            except Exception as exc:
                print(f"[GraspAdjust_2] arm target failed: {exc}; move forward a little")
                vx, last_time = self.x_move_by_depth(block.depth_mm)
                self.dog.move(vx=vx, last_time=last_time, duration=0.5)
                continue

            print(f"[GraspAdjust_2] r={r_mm:.1f}mm limit={GRASP_R_LIMIT_MM:.1f}mm")
            if r_mm > GRASP_R_LIMIT_MM:
                print("[GraspAdjust_2] target too far, move forward a little")
                vx, last_time = self.x_move_by_depth(block.depth_mm)
                self.dog.move(vx=vx, last_time=last_time, duration=0.5)
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
            frame, matches = self.detect_matches(letter)
            if frame is None:
                continue

            if not matches:
                frame, matches = self.refind_target(letter)
                if frame is None or not matches:
                    continue

            matches.sort(key=lambda det: (det.area, det.conf), reverse=True)
            target = matches[0]
            last_seen = target

            _, frame_w = frame.shape[:2]
            # 防止出界，D 中心点偏右时，就认为已经中心对齐了。
            error_x = target.center[0] - frame_w * 0.5 if letter != 'D' else target.center[0] - frame_w * 0.5 - 30
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
        depth_none_count = 0
        target_none_count = 0
        last_depth_mm = None
        last_seen = None

        while time.time() < deadline:
            frame, matches = self.detect_matches(letter)
            if frame is None:
                continue

            # 如果没找到并且给上一次的距离是80cm以内，就直接认为找到了
            if not matches:
                target_none_count += 1
                if target_none_count < 3:
                    print(f"[Box] {letter} not found attempt {target_none_count}/3")
                    time.sleep(0.1)
                    continue
                
                # 连续三次没找到，就站定再识别，防止因为抖动导致的识别失败  
                if last_depth_mm is not None and last_depth_mm < BOX_SEENED_DEPTH_MM:
                    print(
                        f"[Box] {letter} lost for 3 frames after near depth={last_depth_mm:.1f}mm, "
                        "treat as reached"
                    )
                    self.dog.stop()
                    return

                print(f"[Box] {letter} not found after 3 attempts")
                target_none_count = 0
                frame, matches = self.refind_target(letter)
                if frame is None or not matches:
                    continue
            
            # 找到了就清空计数器
            else:
                target_none_count = 0

            matches.sort(key=lambda det: (det.area, det.conf), reverse=True)
            target = matches[0]
            last_seen = target
            if target.depth_mm is not None:
                last_depth_mm = target.depth_mm

            # 连续三次有目标但是没有深度信息就认为到达了
            if target.depth_mm is None:
                depth_none_count += 1
                print(f"[Box] {letter} depth invalid")
                time.sleep(0.1)
                if (
                    depth_none_count >= 3
                    and last_depth_mm is not None
                    and last_depth_mm < BOX_SEENED_DEPTH_MM
                ):
                    print(
                        f"[Box] {letter} depth invalid count={depth_none_count} "
                        f"after near depth={last_depth_mm:.1f}mm, stopped"
                    )
                    self.dog.stop()
                    return
                continue
            
            # 如果距离大于30cm，就往前走一点
            if target.depth_mm >= BOX_LOST_NEAR_DEPTH_MM:
                depth_none_count = 0
                vx, last_time = self.x_move_by_depth(target.depth_mm)
                self.dog.move(vx=vx, last_time=last_time, duration=0.3)
                print(f"[Box] {letter} depth={target.depth_mm:.1f}mm, move forward a little more")
                continue

            self.dog.stop()
            print(f"[Box] reached {letter}")
            return

        raise RuntimeError(f"Failed to reach box {letter}; last_seen={last_seen}")


def main():
    task3 = Task3(status_dict=DASHBOARD_STATUS)
    try:
        task3.start()
        task3.run()
    finally:
        task3.close()


if __name__ == "__main__":
    main()
