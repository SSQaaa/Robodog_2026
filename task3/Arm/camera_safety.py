# -*- coding: utf-8 -*-
"""机械臂逆解候选姿态的相机安全球检查。"""

import math
from dataclasses import dataclass


@dataclass
class CameraSafetyResult:
    safe: bool
    enabled: bool
    min_distance_mm: float = None
    required_distance_mm: float = None
    nearest_segment: str = None
    reason: str = ""


def arm_points(theta1_deg, theta2_deg, theta3_deg, arm_cfg):
    lengths = arm_cfg["link_lengths_mm"]
    l1 = float(lengths["L1"])
    l2 = float(lengths["L2"])
    l3 = float(lengths["L3"])
    grip = float(arm_cfg.get("gripper_offset_mm", 0.0))

    a1 = math.radians(float(theta1_deg))
    a2 = math.radians(float(theta1_deg) + float(theta2_deg))
    a3 = math.radians(float(theta1_deg) + float(theta2_deg) + float(theta3_deg))

    p0 = (0.0, 0.0)
    p1 = (l1 * math.sin(a1), l1 * math.cos(a1))
    p2 = (p1[0] + l2 * math.sin(a2), p1[1] + l2 * math.cos(a2))
    p3 = (p2[0] + l3 * math.sin(a3), p2[1] + l3 * math.cos(a3))
    p4 = (p3[0] + grip * math.sin(a3), p3[1] + grip * math.cos(a3))
    return [p0, p1, p2, p3, p4]


def point_to_segment_distance(point, a, b):
    px, pz = point
    ax, az = a
    bx, bz = b
    abx = bx - ax
    abz = bz - az
    denom = abx * abx + abz * abz
    if denom <= 1e-9:
        return math.hypot(px - ax, pz - az)
    t = ((px - ax) * abx + (pz - az) * abz) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qz = az + t * abz
    return math.hypot(px - qx, pz - qz)


def camera_projection(yaw_deg, arm_cfg):
    safety_cfg = arm_cfg.get("camera_safety", {})
    camera_x, camera_y, camera_z = [float(v) for v in safety_cfg.get("center_base_mm", [150.0, 0.0, 80.0])]
    shoulder_height = float(arm_cfg.get("shoulder_height_mm", 0.0))
    yaw = math.radians(float(yaw_deg))
    camera_r = camera_x * math.cos(yaw) + camera_y * math.sin(yaw)
    camera_side = -camera_x * math.sin(yaw) + camera_y * math.cos(yaw)
    camera_z_plane = camera_z - shoulder_height
    return camera_r, camera_side, camera_z_plane


def check_camera_safety(theta1_deg, theta2_deg, theta3_deg, yaw_deg, arm_cfg):
    safety_cfg = arm_cfg.get("camera_safety", {})
    if not bool(safety_cfg.get("enabled", False)):
        return CameraSafetyResult(safe=True, enabled=False, reason="disabled")

    radius = float(safety_cfg.get("radius_mm", 50.0))
    camera_r, camera_side, camera_z = camera_projection(yaw_deg, arm_cfg)
    if abs(camera_side) >= radius:
        return CameraSafetyResult(safe=True, enabled=True, reason="camera outside yaw plane")

    plane_radius = math.sqrt(max(0.0, radius * radius - camera_side * camera_side))
    camera_point = (camera_r, camera_z)
    points = arm_points(theta1_deg, theta2_deg, theta3_deg, arm_cfg)
    names = ["L1", "L2", "L3", "gripper"]

    distances = []
    for name, a, b in zip(names, points[:-1], points[1:]):
        distances.append((point_to_segment_distance(camera_point, a, b), name))

    min_distance, nearest = min(distances, key=lambda item: item[0])
    safe = min_distance >= plane_radius
    reason = "safe" if safe else f"{nearest} enters camera safety sphere"
    return CameraSafetyResult(
        safe=safe,
        enabled=True,
        min_distance_mm=min_distance,
        required_distance_mm=plane_radius,
        nearest_segment=nearest,
        reason=reason,
    )
