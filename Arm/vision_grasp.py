# -*- coding: utf-8 -*-
"""使用 Orbbec 深度相机检测绿色物块，并可选择执行抓取。"""

import argparse
import time

import cv2
import numpy as np

from config import load_config
from kinematics import print_solution, solve_arm_target
from servo_driver import ServoBus


def start_camera(warmup_s):
    import orbbec_native

    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(float(warmup_s))
    return cam


def get_color_frame(camera):
    frame = camera.get_color_frame()
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8).copy()


def scale_box(box, src_size, dst_size):
    x1, y1, x2, y2 = box
    src_w, src_h = src_size
    dst_w, dst_h = dst_size

    dx1 = int(round(x1 * dst_w / float(src_w)))
    dy1 = int(round(y1 * dst_h / float(src_h)))
    dx2 = int(round(x2 * dst_w / float(src_w)))
    dy2 = int(round(y2 * dst_h / float(src_h)))

    dx1 = max(0, min(dst_w - 1, dx1))
    dx2 = max(0, min(dst_w - 1, dx2))
    dy1 = max(0, min(dst_h - 1, dy1))
    dy2 = max(0, min(dst_h - 1, dy2))

    return dx1, dy1, dx2, dy2


def get_depth_in_color_box(camera, color_box, min_valid_count=20):
    color_w, color_h = camera.get_color_size()
    depth_w, depth_h = camera.get_depth_size()
    depth_box = scale_box(
        color_box,
        src_size=(color_w, color_h),
        dst_size=(depth_w, depth_h),
    )
    depth_mm, valid_count = camera.get_depth_in_box(*depth_box)
    if depth_mm <= 0 or valid_count < int(min_valid_count):
        return None, int(valid_count), depth_box
    return int(depth_mm), int(valid_count), depth_box


def pixel_to_camera(u, v, depth_mm, intrinsics):
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (float(u) - cx) * float(depth_mm) / fx
    y = (float(v) - cy) * float(depth_mm) / fy
    z = float(depth_mm)
    return np.array([x, y, z], dtype=np.float64)


def transform_point(T, point):
    p = np.asarray(point, dtype=np.float64).reshape(3)
    hp = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (np.asarray(T, dtype=np.float64) @ hp)[:3]


def wait_for_grasp_result(camera, config, max_attempts=60, retry_delay_s=0.05):
    """Wait for a color frame whose matching depth data is ready."""
    last_error = None
    for attempt in range(1, int(max_attempts) + 1):
        frame = get_color_frame(camera)
        if frame is None:
            last_error = RuntimeError("Color frame is not ready")
        else:
            try:
                return frame, compute_grasp(camera, frame, config)
            except RuntimeError as exc:
                message = str(exc)
                if not (
                    message.startswith("Green block not detected")
                    or message.startswith("No valid depth in block box")
                ):
                    raise
                last_error = exc

        if attempt == 1 or attempt % 10 == 0:
            print(f"[Wait] grasp frame {attempt}/{max_attempts} not ready: {last_error}")
        time.sleep(float(retry_delay_s))

    raise RuntimeError(
        f"No usable color/depth frame after {max_attempts} attempts; last error: {last_error}"
    )


def detect_green_block(frame_bgr, block_cfg):
    lower = np.asarray(block_cfg.get("hsv_lower", [35, 40, 40]), dtype=np.uint8)
    upper = np.asarray(block_cfg.get("hsv_upper", [85, 255, 255]), dtype=np.uint8)
    min_area = float(block_cfg.get("min_area_px", 500.0))
    kernel_size = max(1, int(block_cfg.get("morph_kernel_px", 5)))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None, mask
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(contour)
    return {
        "center": (float(cx), float(cy)),
        "area": float(cv2.contourArea(contour)),
        "angle_deg": float(angle),
        "rect_size": (float(w), float(h)),
        "box": box,
        "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
    }, mask


def draw_block(frame, detection, result=None):
    if detection is None:
        return frame
    cv2.drawContours(frame, [detection["box"]], 0, (0, 255, 0), 2)
    cx, cy = detection["center"]
    cv2.circle(frame, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
    label = "green block"
    if result is not None:
        label += f" {result['depth_mm'] / 1000.0:.2f}m"
    x1, y1, _, _ = detection["bbox"]
    cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def compute_grasp(camera, frame, config):
    block_cfg = config["block"]
    detection, mask = detect_green_block(frame, block_cfg)
    if detection is None:
        raise RuntimeError("Green block not detected. Adjust HSV/min_area or lighting.")

    color_intrinsics = camera.get_color_intrinsics()
    if config["camera"].get("T_base_camera") is None:
        raise RuntimeError("camera.T_base_camera is missing in params.json. Run calibration.py --samples 60 --save first.")
    T_base_camera = np.asarray(config["camera"]["T_base_camera"], dtype=np.float64)
    u, v = detection["center"]
    depth_mm, valid_count, depth_box = get_depth_in_color_box(
        camera,
        detection["bbox"],
        min_valid_count=int(block_cfg.get("min_valid_depth_count", 20)),
    )
    if depth_mm is None:
        raise RuntimeError(
            f"No valid depth in block box, valid_count={valid_count}, "
            f"color_box={detection['bbox']}, depth_box={depth_box}"
        )

    point_camera = pixel_to_camera(u, v, depth_mm, color_intrinsics)
    measured_base = transform_point(T_base_camera, point_camera)

    size = np.asarray(block_cfg.get("size_mm", [100.0, 50.0, 50.0]), dtype=np.float64)
    grasp_base = measured_base.copy()
    grasp_base[2] = (
        float(block_cfg.get("table_z_base_mm", 0.0))
        + float(size[2]) * 0.5
        + float(block_cfg.get("grasp_z_offset_mm", 0.0))
    )
    grasp_base += np.asarray(block_cfg.get("grasp_offset_base_mm", [0.0, 0.0, 0.0]), dtype=np.float64)

    try:
        solution = solve_arm_target(grasp_base[0], grasp_base[1], grasp_base[2], config["arm"])
    except ValueError as exc:
        raise ValueError(
            "IK failed for detected block: "
            f"pixel=({u:.1f}, {v:.1f}), depth={int(depth_mm)}, valid={int(valid_count)}, "
            f"color_box={detection['bbox']}, depth_box={depth_box}, "
            f"camera_xyz=({point_camera[0]:.1f}, {point_camera[1]:.1f}, {point_camera[2]:.1f}), "
            f"base_xyz=({measured_base[0]:.1f}, {measured_base[1]:.1f}, {measured_base[2]:.1f}), "
            f"grasp_xyz=({grasp_base[0]:.1f}, {grasp_base[1]:.1f}, {grasp_base[2]:.1f}); {exc}"
        ) from exc
    return {
        "detection": detection,
        "mask": mask,
        "pixel": (float(u), float(v)),
        "depth_mm": int(depth_mm),
        "valid_count": int(valid_count),
        "depth_box": depth_box,
        "point_camera": point_camera,
        "measured_base": measured_base,
        "grasp_base": grasp_base,
        "solution": solution,
    }


def print_result(result):
    cam = result["point_camera"]
    raw = result["measured_base"]
    grasp = result["grasp_base"]
    print("[Calibration] T_base_camera source: saved")
    print(f"[Block] pixel=({result['pixel'][0]:.1f}, {result['pixel'][1]:.1f}) depth={result['depth_mm']} valid={result['valid_count']}")
    print(f"[Depth] color_box={result['detection']['bbox']} depth_box={result['depth_box']}")
    print(f"[Camera] xyz=({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f})")
    print(f"[Base] measured xyz=({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f})")
    print(f"[Base] grasp xyz=({grasp[0]:.1f}, {grasp[1]:.1f}, {grasp[2]:.1f})")
    print_solution(result["solution"])


def read_solution_status(config, solution):
    try:
        bus = ServoBus(config["arm"])
        try:
            ids = sorted(solution.servo_targets.keys())
            return {servo_id: bus.read_status(servo_id) for servo_id in ids}
        finally:
            bus.close()
    except Exception as exc:
        print(f"[DryRun] current pos read failed: {exc}")
        return None


def print_grasp_dry_run(result, config):
    arm_cfg = config["arm"]
    lift = float(arm_cfg.get("pre_grasp_lift_mm", 40.0))
    grasp = result["grasp_base"]
    pre_solution = solve_arm_target(grasp[0], grasp[1], grasp[2] + lift, arm_cfg)
    grasp_solution = result["solution"]

    print("[DryRun] pre-grasp arm angles and servo targets")
    pre_status = read_solution_status(config, pre_solution)
    print_solution(pre_solution, current_status=pre_status)

    print("[DryRun] grasp arm angles and servo targets")
    grasp_status = read_solution_status(config, grasp_solution)
    print_solution(grasp_solution, current_status=grasp_status)


def execute_grasp(result, config):
    arm_cfg = config["arm"]
    solution = result["solution"]
    lift = float(arm_cfg.get("pre_grasp_lift_mm", 40.0))
    grasp = result["grasp_base"]
    pre_solution = solve_arm_target(grasp[0], grasp[1], grasp[2] + lift, arm_cfg)

    bus = ServoBus(arm_cfg)
    try:
        print("[Execute] current before motion")
        bus.print_status()
        print("[Execute] open gripper")
        bus.open_gripper()
        print("[Execute] move pre-grasp")
        print_solution(pre_solution)
        bus.move_targets(pre_solution.servo_targets, wait_s=1.5)
        print("[Execute] move grasp")
        print_solution(solution)
        bus.move_targets(solution.servo_targets, wait_s=1.5)
        print("[Execute] close gripper with current limit")
        bus.close_gripper_protected()
        print("[Execute] lift")
        bus.move_targets(pre_solution.servo_targets, wait_s=1.5)
        print("[Execute] final status")
        bus.print_status()
    finally:
        bus.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and grasp a green block.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print target angles/positions without moving servos.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=60, help="Maximum color/depth frame attempts.")
    parser.add_argument("--retry-delay", type=float, default=0.05, help="Delay between frame attempts in seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run and args.execute:
        raise ValueError("--dry-run and --execute cannot be used together")
    config = load_config(args.config)
    camera = start_camera(config["camera"].get("warmup_s", 1.0))
    try:
        print(f"[Orbbec] color size: {camera.get_color_size()}")
        print(f"[Orbbec] depth size : {camera.get_depth_size()}")
        frame, result = wait_for_grasp_result(
            camera,
            config,
            max_attempts=args.max_attempts,
            retry_delay_s=args.retry_delay,
        )
        print_result(result)
        if args.show:
            draw_block(frame, result["detection"], result)
            cv2.imshow("task3_new grasp", frame)
            cv2.imshow("task3_new green mask", result["mask"])
            cv2.waitKey(0)
        if args.execute:
            execute_grasp(result, config)
        else:
            print_grasp_dry_run(result, config)
            print("[DryRun] not moving servos. Add --execute to run the grasp sequence.")
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
