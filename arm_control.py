# -*- coding: utf-8 -*-
"""Task3 arm control used by the top-level mission flow."""

import math
import os
import sys
import time

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARM_DIR = os.path.join(BASE_DIR, "Arm")
if ARM_DIR not in sys.path:
    sys.path.insert(0, ARM_DIR)

from config import DEFAULT_CONFIG_PATH, load_config
from coordinates import pixel_to_camera, transform_point
from kinematics import print_solution, solve_arm_target
from servo_driver import ServoBus


class ArmControl:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.arm_cfg = self.config["arm"]
        self.camera_cfg = self.config["camera"]
        self.block_cfg = self.config["block"]
        self.bus = None

        if self.camera_cfg.get("T_base_camera") is None:
            raise RuntimeError("camera.T_base_camera missing in Arm/params.json; run calibration first.")
        self.T_base_camera = np.asarray(self.camera_cfg["T_base_camera"], dtype=np.float64)

    def start(self):
        self.bus = ServoBus(self.arm_cfg)
        return self

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def reset(self):
        pose = {int(k): int(v) for k, v in self.arm_cfg.get("reset_pose", {}).items()}
        if not pose:
            print("[Arm] reset_pose missing, skip reset")
            return
        print("[Arm] reset to configured pose")
        self.bus.move_targets(pose, wait_s=2.0)

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

        pre_solution = plan["pre_solution"]
        post_solution = plan["post_solution"]

        self.bus.open_gripper()
        self.bus.move_targets(pre_solution.servo_targets, wait_s=1.5)
        self.bus.move_targets(result["solution"].servo_targets, wait_s=1.5)
        time.sleep(float(self.arm_cfg.get("grasp_settle_s", 0.3)))
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

    def lift_after_failed_grasp(self, detection, color_intrinsics, lift_mm=100.0):
        """Lift away from the failed grasp point; horizontal retraction is allowed."""
        result = self.compute_grasp_target(detection, color_intrinsics)
        solution = self.solve_lift_target(result["grasp_base"], lift_mm, "failed_grasp")
        print(
            f"[PickRetry] lift after failed grasp: "
            f"z={result['grasp_base'][2]:.1f}->{solution.z_mm:.1f}mm, "
            f"r={result['solution'].r_mm:.1f}->{solution.r_mm:.1f}mm"
        )
        self.bus.move_targets(solution.servo_targets, wait_s=1.5)

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

    def place_block(self):
        print("[Place] keep post-grasp pose and open gripper")
        self.bus.open_gripper()
        time.sleep(0.5)

    def compute_place_pose(self):
        table_z = float(self.block_cfg.get("table_z_base_mm", 0.0))
        place_z = table_z + float(self.arm_cfg.get("place_height_above_table_mm", 27.0))
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


def reset_arm(config_path=DEFAULT_CONFIG_PATH):
    arm = ArmControl(config_path=config_path).start()
    try:
        arm.reset()
    finally:
        arm.close()
