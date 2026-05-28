# -*- coding: utf-8 -*-
"""机械臂控制模块：使用最新 Arm 目录中的参数和驱动。"""

import os
import sys
import time

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARM_DIR = os.path.join(BASE_DIR, "Arm")
if ARM_DIR not in sys.path:
    sys.path.insert(0, ARM_DIR)

from config import DEFAULT_CONFIG_PATH, load_config
from kinematics import print_solution, solve_arm_target
from servo_driver import ServoBus
from vision_grasp import pixel_to_camera, transform_point


class ArmControl:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH, dry_run=False):
        self.config = load_config(config_path)
        self.arm_cfg = self.config["arm"]
        self.camera_cfg = self.config["camera"]
        self.block_cfg = self.config["block"]
        self.dry_run = bool(dry_run)
        self.bus = None

        if self.camera_cfg.get("T_base_camera") is None:
            raise RuntimeError("camera.T_base_camera missing in Arm/params.json; run calibration first.")
        self.T_base_camera = np.asarray(self.camera_cfg["T_base_camera"], dtype=np.float64)

    # 只在实车模式下打开舵机总线。
    def start(self):
        if self.dry_run:
            print("[DryRun][Arm] servo bus not opened")
            return self
        self.bus = ServoBus(self.arm_cfg)
        return self

    def close(self):
        if self.bus is not None:
            self.bus.close()

    # 将视觉检测结果转换为机械臂目标点，并执行抓取。
    def pick_block(self, block_class, detection, color_intrinsics):
        if detection is None:
            raise RuntimeError(f"{block_class} block not detected with valid depth")
        if detection.depth_mm is None:
            raise RuntimeError(f"{block_class} block has no valid depth")
        while True:
            result = self.compute_grasp_target(detection, color_intrinsics)
            print(
                "[Pick] {} pixel=({:.1f},{:.1f}) depth={} valid={}".format(
                    block_class,
                    detection.center[0],
                    detection.center[1],
                    detection.depth_mm,
                    detection.valid_count,
                )
            )
            print_solution(result["solution"])

            if self.dry_run:
                print("[DryRun][Arm] grasp sequence skipped")
                return

            lift = float(self.arm_cfg.get("pre_grasp_lift_mm", 40.0))
            grasp = result["grasp_base"]
            pre_solution = solve_arm_target(grasp[0], grasp[1], grasp[2] + lift, self.arm_cfg)

            self.bus.open_gripper()
            self.bus.move_targets(pre_solution.servo_targets, wait_s=1.5)
            self.bus.move_targets(result["solution"].servo_targets, wait_s=1.5)
            self.bus.close_gripper_protected()
            self.bus.move_targets(pre_solution.servo_targets, wait_s=1.5)

            gripper_I = self.bus.read_status(1).current_units
            print(f"[Pick] gripper current={gripper_I} units")
            if gripper_I is not None and gripper_I >= 6:
                print("[Pick] Successfully grasped the block")
                break
            else:
                print("[Pick] Warning:failed to grasp the block, retrying...")
                continue

    # 松开抓手放下物块，具体时机由 task3.py 决定。
    def place_block(self):
        print("[Place] move base servo to center position")
        if self.dry_run:
            print("[DryRun][Arm] base servo move skipped")
        else:
            base_center = self.arm_cfg["servos"]["6"]["zero"]
            self.bus.write_pos(6, base_center)
            time.sleep(1.0)

        print("[Place] open gripper")
        if self.dry_run:
            print("[DryRun][Arm] gripper open skipped")
            return
        self.bus.open_gripper()
        time.sleep(0.5)

    # 使用已保存的相机标定矩阵，把 YOLO 目标转换到机械臂基座坐标。
    def compute_grasp_target(self, detection, color_intrinsics):
        u, v = detection.center
        point_camera = pixel_to_camera(u, v, detection.depth_mm, color_intrinsics)
        measured_base = transform_point(self.T_base_camera, point_camera)

        size = np.asarray(self.block_cfg.get("size_mm", [100.0, 50.0, 50.0]), dtype=np.float64)
        grasp_base = measured_base.copy()
        grasp_base[2] = (
            float(self.block_cfg.get("table_z_base_mm", 0.0))
            + float(size[2]) * 0.5
            + float(self.block_cfg.get("grasp_z_offset_mm", 0.0))
        )
        grasp_base += np.asarray(self.block_cfg.get("grasp_offset_base_mm", [0.0, 0.0, 0.0]), dtype=np.float64)

        solution = solve_arm_target(grasp_base[0], grasp_base[1], grasp_base[2], self.arm_cfg)
        return {
            "point_camera": point_camera,
            "measured_base": measured_base,
            "grasp_base": grasp_base,
            "solution": solution,
        }
