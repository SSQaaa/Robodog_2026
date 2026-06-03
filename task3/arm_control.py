# -*- coding: utf-8 -*-
"""机械臂控制模块：负责坐标解算、舵机动作和抓手电流判断。"""

import os
import sys
import time
import math

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

    # 打开舵机总线。
    def start(self):
        if self.dry_run:
            print("[DryRun][Arm] servo bus not opened")
            return self
        self.bus = ServoBus(self.arm_cfg)
        return self

    def close(self):
        if self.bus is not None:
            self.bus.close()

    # 根据当前视觉结果执行一次抓取，返回是否抓取成功。
    def pick_block(self, block_class, detection, color_intrinsics):
        if detection is None:
            raise RuntimeError(f"{block_class} block not detected with valid depth")
        if detection.depth_mm is None:
            raise RuntimeError(f"{block_class} block has no valid depth")

        try:
            plan = self.compute_pick_plan(detection, color_intrinsics)
        except Exception as exc:
            print(f"[Pick] arm target failed before motion: {exc}")
            return False

        result = plan["target"]
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
            return True

        pre_solution = plan["pre_solution"]
        post_solution = plan["post_solution"]

        self.bus.open_gripper()
        self.bus.move_targets(pre_solution.servo_targets, wait_s=1.5)
        self.bus.move_targets(result["solution"].servo_targets, wait_s=1.5)
        self.bus.close_gripper_protected()
        self.bus.move_targets(post_solution.servo_targets, wait_s=1.5)

        gripper_status = self.bus.read_status(self.arm_cfg["ids"]["gripper"])
        gripper_i = gripper_status.current_units
        gripper_overload = gripper_status.error is not None and "pos_error=32" in gripper_status.error
        print(f"[Pick] gripper current={gripper_i} units error={gripper_status.error}")
        if (gripper_i is not None and gripper_i > 10) or gripper_overload:
            print("[Pick] Successfully grasped the block")
            return True

        print("[Pick] Warning: failed to grasp the block")
        return False

    def compute_pick_plan(self, detection, color_intrinsics):
        result = self.compute_grasp_target(detection, color_intrinsics)
        pre_lift = float(self.arm_cfg.get("pre_grasp_lift_mm", 40.0))
        post_lift = float(self.arm_cfg.get("post_grasp_lift_mm", pre_lift))
        grasp = result["grasp_base"]
        pre_solution = self.solve_lift_target(grasp, pre_lift, "pre")
        post_solution = self.solve_lift_target(grasp, post_lift, "post")
        return {
            "target": result,
            "pre_solution": pre_solution,
            "post_solution": post_solution,
        }

    def solve_lift_target(self, grasp_base, lift_mm, label):
        x, y, z = [float(value) for value in grasp_base]
        target_z = z + float(lift_mm)
        yaw = math.atan2(y, x)
        original_r = math.hypot(x, y)

        try:
            return solve_arm_target(x, y, target_z, self.arm_cfg)
        except ValueError as original_exc:
            last_exc = original_exc

        r = original_r - 5.0
        while r > 1.0:
            candidate_x = r * math.cos(yaw)
            candidate_y = r * math.sin(yaw)
            try:
                solution = solve_arm_target(candidate_x, candidate_y, target_z, self.arm_cfg)
                print(
                    f"[IK] {label}_lift retracted r {original_r:.1f}->{solution.r_mm:.1f}mm "
                    f"at z={target_z:.1f}mm"
                )
                return solution
            except ValueError as exc:
                last_exc = exc
                r -= 5.0

        raise ValueError(f"{label}_lift target is unreachable even with retract: {last_exc}")

    # 松开抓手放下物块，具体时机由 task3.py 决定。
    def place_block(self):
        solution = self.compute_place_pose()
        print("[Place] move to forward reach pose")
        print_solution(solution)
        if self.dry_run:
            print("[DryRun][Arm] place pose move skipped")
        else:
            self.bus.move_targets(solution.servo_targets, wait_s=1.5)

        print("[Place] open gripper")
        if self.dry_run:
            print("[DryRun][Arm] gripper open skipped")
            return
        self.bus.open_gripper()
        time.sleep(0.5)

    def compute_place_pose(self):
        table_z = float(self.block_cfg.get("table_z_base_mm", 0.0))
        place_z = table_z + float(self.arm_cfg.get("place_height_above_table_mm", 100.0))
        lengths = self.arm_cfg["link_lengths_mm"]
        l1 = float(lengths["L1"])
        l2 = float(lengths["L2"])
        l3 = float(lengths["L3"])
        gripper_offset = float(self.arm_cfg.get("gripper_offset_mm", 0.0))
        shoulder_height = float(self.arm_cfg.get("shoulder_height_mm", 0.0))
        wrist_z = place_z - shoulder_height
        max_wrist_r = math.sqrt(max(0.0, (l1 + l2) * (l1 + l2) - wrist_z * wrist_z))
        r = max_wrist_r + l3 + gripper_offset - 1.0

        last_exc = None
        while r > 1.0:
            try:
                return solve_arm_target(r, 0.0, place_z, self.arm_cfg)
            except ValueError as exc:
                last_exc = exc
                r -= 5.0

        raise ValueError(f"Forward place pose is unreachable: {last_exc}")

    # 把 YOLO 像素点和深度转换成机械臂基座坐标，并求逆解。
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
