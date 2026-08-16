# -*- coding: utf-8 -*-
"""使用 Orbbec 深度相机检测绿色物块，并可选择执行抓取。"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from coordinates import pixel_to_camera, transform_point
from kinematics import print_solution
from servo_driver import ServoBus


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from arm_control import ArmControl


@dataclass
class StandaloneDetection:
    """Detection fields consumed by the same ArmControl used by Task3."""

    center: tuple
    depth_mm: int
    valid_count: int


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


def wait_for_grasp_result(camera, arm, max_attempts=60, retry_delay_s=0.05):
    """Wait for a color frame whose matching depth data is ready."""
    last_error = None
    for attempt in range(1, int(max_attempts) + 1):
        frame = get_color_frame(camera)
        if frame is None:
            last_error = RuntimeError("Color frame is not ready")
        else:
            try:
                return frame, compute_grasp(camera, frame, arm)
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


def compute_grasp(camera, frame, arm):
    block_cfg = arm.block_cfg
    detection, mask = detect_green_block(frame, block_cfg)
    if detection is None:
        raise RuntimeError("Green block not detected. Adjust HSV/min_area or lighting.")

    color_intrinsics = camera.get_color_intrinsics()
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

    task3_detection = StandaloneDetection(
        center=(float(u), float(v)),
        depth_mm=int(depth_mm),
        valid_count=int(valid_count),
    )
    try:
        plan = arm.compute_pick_plan(task3_detection, color_intrinsics)
    except ValueError as exc:
        point_camera = pixel_to_camera(u, v, depth_mm, color_intrinsics)
        measured_base = transform_point(arm.T_base_camera, point_camera)
        raise ValueError(
            "IK failed for detected block: "
            f"pixel=({u:.1f}, {v:.1f}), depth={int(depth_mm)}, valid={int(valid_count)}, "
            f"color_box={detection['bbox']}, depth_box={depth_box}, "
            f"camera_xyz=({point_camera[0]:.1f}, {point_camera[1]:.1f}, {point_camera[2]:.1f}), "
            f"base_xyz=({measured_base[0]:.1f}, {measured_base[1]:.1f}, {measured_base[2]:.1f}); {exc}"
        ) from exc
    result = dict(plan["target"])
    result.update({
        "detection": detection,
        "mask": mask,
        "pixel": (float(u), float(v)),
        "depth_mm": int(depth_mm),
        "valid_count": int(valid_count),
        "depth_box": depth_box,
        "plan": plan,
        "task3_detection": task3_detection,
        "color_intrinsics": color_intrinsics,
    })
    return result


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
    plan = result["plan"]
    pre_solution = plan["pre_solution"]
    grasp_solution = plan["target"]["solution"]
    post_solution = plan["post_solution"]

    print("[DryRun] pre-grasp arm angles and servo targets")
    pre_status = read_solution_status(config, pre_solution)
    print_solution(pre_solution, current_status=pre_status)

    print("[DryRun] grasp arm angles and servo targets")
    grasp_status = read_solution_status(config, grasp_solution)
    print_solution(grasp_solution, current_status=grasp_status)

    print("[DryRun] post-grasp arm angles and servo targets")
    post_status = read_solution_status(config, post_solution)
    print_solution(post_solution, current_status=post_status)


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
    arm = ArmControl(config_path=args.config)
    config = arm.config
    camera = start_camera(config["camera"].get("warmup_s", 1.0))
    try:
        print(f"[Orbbec] color size: {camera.get_color_size()}")
        print(f"[Orbbec] depth size : {camera.get_depth_size()}")
        frame, result = wait_for_grasp_result(
            camera,
            arm,
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
            arm.start()
            try:
                picked = arm.pick_block(
                    "Green",
                    result["task3_detection"],
                    result["color_intrinsics"],
                )
                print(f"[Execute] Task3 pick result={picked}")
            finally:
                arm.close()
        else:
            print_grasp_dry_run(result, config)
            print("[DryRun] not moving servos. Add --execute to run the grasp sequence.")
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
