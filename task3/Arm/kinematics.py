# -*- coding: utf-8 -*-
"""任务三机械臂平面逆运动学。

关节角约定：
- theta1：连杆 1 相对竖直方向的角度，顺时针为正。
- theta2：连杆 2 相对连杆 1 的角度，顺时针为正。
- theta3：连杆 3 相对连杆 2 的角度，顺时针为正。
- 连杆 3 和抓手水平时，theta1 + theta2 + theta3 = 90 度。
"""

import argparse
import math
from dataclasses import dataclass

from camera_safety import check_camera_safety
from config import load_config


@dataclass
class ArmSolution:
    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float
    r_mm: float
    wrist_r_mm: float
    wrist_z_mm: float
    theta1_deg: float
    theta2_deg: float
    theta3_deg: float
    servo_targets: dict


def link_points(theta1_deg, theta2_deg, theta3_deg, arm_cfg, include_gripper=True):
    lengths = arm_cfg["link_lengths_mm"]
    l1 = float(lengths["L1"])
    l2 = float(lengths["L2"])
    l3 = float(lengths["L3"])
    gripper_offset = float(arm_cfg.get("gripper_offset_mm", 0.0)) if include_gripper else 0.0

    a1 = math.radians(float(theta1_deg))
    a2 = math.radians(float(theta1_deg) + float(theta2_deg))
    a3 = math.radians(float(theta1_deg) + float(theta2_deg) + float(theta3_deg))

    p0 = (0.0, 0.0)
    p1 = (p0[0] + l1 * math.sin(a1), p0[1] + l1 * math.cos(a1))
    p2 = (p1[0] + l2 * math.sin(a2), p1[1] + l2 * math.cos(a2))
    p3 = (p2[0] + l3 * math.sin(a3), p2[1] + l3 * math.cos(a3))
    p4 = (p3[0] + gripper_offset * math.sin(a3), p3[1] + gripper_offset * math.cos(a3))
    return [p0, p1, p2, p3, p4]


def show_arm_pose(theta1_deg, theta2_deg, theta3_deg, arm_cfg, title=None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to draw the arm pose. Install it on the machine that runs plotting.") from exc

    points = link_points(theta1_deg, theta2_deg, theta3_deg, arm_cfg)
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    labels = ["shoulder", "link1", "link2", "link3", "grasp"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs[:4], zs[:4], "-o", linewidth=4, markersize=8, label="links")
    ax.plot(xs[3:], zs[3:], "--o", linewidth=3, markersize=7, label="gripper offset")
    for label, x, z in zip(labels, xs, zs):
        ax.text(x + 4, z + 4, label, fontsize=9)

    ax.axhline(0, color="0.75", linewidth=1)
    ax.axvline(0, color="0.75", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("forward r (mm)")
    ax.set_ylabel("height from shoulder (mm)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    pad = 40.0
    min_x = min(xs + [0.0]) - pad
    max_x = max(xs + [0.0]) + pad
    min_z = min(zs + [0.0]) - pad
    max_z = max(zs + [0.0]) + pad
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)

    plt.show()
    plt.close(fig)


def _servo_pos(arm_cfg, servo_id, angle_deg):
    cfg = arm_cfg["servos"][str(int(servo_id))]
    ticks_per_degree = float(arm_cfg.get("ticks_per_degree", 4096.0 / 360.0))
    raw = float(cfg["zero"]) + float(cfg.get("direction", 1)) * float(angle_deg) * ticks_per_degree
    return int(round(max(int(cfg.get("min", 0)), min(int(cfg.get("max", 4095)), raw))))


def _limit_margin(arm_cfg, targets):
    margins = []
    for servo_id, pos in targets.items():
        cfg = arm_cfg["servos"].get(str(int(servo_id)), {})
        margins.append(min(int(pos) - int(cfg.get("min", 0)), int(cfg.get("max", 4095)) - int(pos)))
    return min(margins) if margins else 0


def solve_arm_target(x_mm, y_mm, z_mm, arm_cfg):
    ids = arm_cfg["ids"]
    lengths = arm_cfg["link_lengths_mm"]
    l1 = float(lengths["L1"])
    l2 = float(lengths["L2"])
    l3 = float(lengths["L3"])
    gripper_offset = float(arm_cfg.get("gripper_offset_mm", 0.0))
    shoulder_height = float(arm_cfg.get("shoulder_height_mm", 0.0))
    horizontal_sum = float(arm_cfg.get("horizontal_sum_deg", 90.0))

    x = float(x_mm)
    y = float(y_mm)
    z = float(z_mm)
    yaw_deg = math.degrees(math.atan2(y, x))
    r = math.hypot(x, y)

    wrist_r = r - gripper_offset - l3
    wrist_z = z - shoulder_height
    if wrist_r < 0:
        raise ValueError(
            f"Target is too close after gripper/link3 offset: r={r:.1f}, "
            f"offset={gripper_offset + l3:.1f}, wrist_r={wrist_r:.1f}"
        )

    d2 = wrist_r * wrist_r + wrist_z * wrist_z
    d = math.sqrt(d2)
    if d > l1 + l2 or d < abs(l1 - l2):
        raise ValueError(
            f"Target wrist point is unreachable: wrist_r={wrist_r:.1f}, wrist_z={wrist_z:.1f}, "
            f"distance={d:.1f}, reach=[{abs(l1 - l2):.1f}, {l1 + l2:.1f}]"
        )

    cos_q2 = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    candidates = []
    rejected = []
    for q2_standard in (math.acos(cos_q2), -math.acos(cos_q2)):
        phi1 = math.atan2(wrist_z, wrist_r) - math.atan2(l2 * math.sin(q2_standard), l1 + l2 * math.cos(q2_standard))
        phi2 = phi1 + q2_standard

        # 将标准水平参考角转换为本项目的竖直参考、顺时针为正的角度约定。
        theta1 = math.degrees((math.pi / 2.0) - phi1)
        link2_abs = math.degrees((math.pi / 2.0) - phi2)
        theta2 = link2_abs - theta1
        theta3 = horizontal_sum - theta1 - theta2

        targets = {
            ids["base"]: _servo_pos(arm_cfg, ids["base"], yaw_deg),
            ids["shoulder"]: _servo_pos(arm_cfg, ids["shoulder"], theta1),
            ids["elbow"]: _servo_pos(arm_cfg, ids["elbow"], theta2),
            ids["wrist"]: _servo_pos(arm_cfg, ids["wrist"], theta3),
        }
        safety = check_camera_safety(theta1, theta2, theta3, yaw_deg, arm_cfg)
        if not safety.safe:
            rejected.append(
                "camera safety rejected theta=({:.1f}, {:.1f}, {:.1f}): {}, "
                "distance={:.1f}, required={:.1f}".format(
                    theta1,
                    theta2,
                    theta3,
                    safety.reason,
                    safety.min_distance_mm,
                    safety.required_distance_mm,
                )
            )
            continue
        margin = _limit_margin(arm_cfg, targets)
        weights = arm_cfg.get("ik_score_weights", {})
        theta1_weight = float(weights.get("theta1", 0.3))
        theta2_weight = float(weights.get("theta2", 0.3))
        theta3_weight = float(weights.get("theta3_to_90", 0.1))
        posture_penalty = (
            theta1_weight * abs(theta1)
            + theta2_weight * abs(theta2)
            + theta3_weight * abs(theta3 - 90.0)
        )
        score = margin - posture_penalty
        candidates.append((score, theta1, theta2, theta3, targets))

    if not candidates:
        detail = "; ".join(rejected) if rejected else "no valid IK candidates"
        raise ValueError(f"No valid IK candidate after camera safety filtering: {detail}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, theta1, theta2, theta3, targets = candidates[0]
    return ArmSolution(
        x_mm=x,
        y_mm=y,
        z_mm=z,
        yaw_deg=yaw_deg,
        r_mm=r,
        wrist_r_mm=wrist_r,
        wrist_z_mm=wrist_z,
        theta1_deg=theta1,
        theta2_deg=theta2,
        theta3_deg=theta3,
        servo_targets=targets,
    )


def print_solution(solution, current_status=None):
    print(
        "[IK] target xyz=({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} r={:.1f} "
        "wrist=({:.1f}, {:.1f})".format(
            solution.x_mm,
            solution.y_mm,
            solution.z_mm,
            solution.yaw_deg,
            solution.r_mm,
            solution.wrist_r_mm,
            solution.wrist_z_mm,
        )
    )
    print(
        "[IK] theta1={:.2f}, theta2={:.2f}, theta3={:.2f}, sum={:.2f}".format(
            solution.theta1_deg,
            solution.theta2_deg,
            solution.theta3_deg,
            solution.theta1_deg + solution.theta2_deg + solution.theta3_deg,
        )
    )
    print(f"[IK] servo targets: {solution.servo_targets}")
    if current_status:
        for servo_id, status in current_status.items():
            print(f"[IK] current id={servo_id} pos={status.position} current={status.current_units} error={status.error}")
    else:
        print("[IK] current pos: not available")


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run task3_new arm IK.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--theta1", type=float, help="Plot an explicit theta1 instead of solving IK.")
    parser.add_argument("--theta2", type=float, help="Plot an explicit theta2 instead of solving IK.")
    parser.add_argument("--theta3", type=float, help="Plot an explicit theta3 instead of solving IK.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    explicit_theta = args.theta1 is not None or args.theta2 is not None or args.theta3 is not None
    if explicit_theta:
        if args.theta1 is None or args.theta2 is None or args.theta3 is None:
            raise ValueError("--theta1, --theta2, and --theta3 must be provided together")
        print(
            "[IK] explicit theta1={:.2f}, theta2={:.2f}, theta3={:.2f}, sum={:.2f}".format(
                args.theta1,
                args.theta2,
                args.theta3,
                args.theta1 + args.theta2 + args.theta3,
            )
        )
        safety = check_camera_safety(args.theta1, args.theta2, args.theta3, 0.0, config["arm"])
        print(
            "[Safety] safe={} enabled={} segment={} distance={} required={} reason={}".format(
                safety.safe,
                safety.enabled,
                safety.nearest_segment,
                None if safety.min_distance_mm is None else round(safety.min_distance_mm, 1),
                None if safety.required_distance_mm is None else round(safety.required_distance_mm, 1),
                safety.reason,
            )
        )
        show_arm_pose(args.theta1, args.theta2, args.theta3, config["arm"])
        return

    if args.x is None or args.y is None or args.z is None:
        raise ValueError("--x, --y, and --z are required unless plotting explicit --theta1/--theta2/--theta3")

    solution = solve_arm_target(args.x, args.y, args.z, config["arm"])
    current_status = None
    try:
        from servo_driver import ServoBus
        bus = ServoBus(config["arm"])
        try:
            ids = sorted(solution.servo_targets.keys())
            current_status = {servo_id: bus.read_status(servo_id) for servo_id in ids}
        finally:
            bus.close()
    except Exception as exc:
        print(f"[IK] current pos: failed to read servos: {exc}")
    print_solution(solution, current_status=current_status)
    show_arm_pose(
        solution.theta1_deg,
        solution.theta2_deg,
        solution.theta3_deg,
        config["arm"],
        title="IK pose for xyz=({:.1f}, {:.1f}, {:.1f})".format(args.x, args.y, args.z),
    )


if __name__ == "__main__":
    main()
