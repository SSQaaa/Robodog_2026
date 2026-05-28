# -*- coding: utf-8 -*-
"""使用 AprilTag id=0 标定相机到机械臂基座的变换。

本文件保留标定所需的小型坐标变换和标签检测工具，方便独立运行标定流程。
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import load_config, save_config


@dataclass
class TagDetection:
    tag_id: int
    center: tuple
    corners: np.ndarray
    T_camera_tag: np.ndarray


def as_transform(value):
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Transform must be 4x4")
    return T


def fixed_tag_camera_calibration(T_base_tag, T_camera_tag):
    return as_transform(T_base_tag) @ np.linalg.inv(as_transform(T_camera_tag))


def average_transforms(transforms):
    mats = np.asarray([as_transform(T) for T in transforms], dtype=np.float64)
    t_mean = np.mean(mats[:, :3, 3], axis=0)
    R_mean = np.mean(mats[:, :3, :3], axis=0)
    u, _, vt = np.linalg.svd(R_mean)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t_mean
    return out


def filter_transform_samples(transforms, max_translation_error_mm):
    if not transforms:
        return [], None, None
    mats = np.asarray([as_transform(T) for T in transforms], dtype=np.float64)
    translations = mats[:, :3, 3]
    median_t = np.median(translations, axis=0)
    errors = np.linalg.norm(translations - median_t, axis=1)
    kept = [transforms[i] for i, err in enumerate(errors) if err <= float(max_translation_error_mm)]
    if not kept:
        kept = list(transforms)
    kept_t = np.asarray([as_transform(T)[:3, 3] for T in kept], dtype=np.float64)
    return kept, np.mean(kept_t, axis=0), np.std(kept_t, axis=0)


def detect_tags(frame_bgr, intrinsics, tag_size_mm, family="tag36h11"):
    import apriltag

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    detector = apriltag.Detector(apriltag.DetectorOptions(families=str(family)))
    raw = detector.detect(gray)
    camera_params = (
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )
    detections = []
    for det in raw:
        pose, _, _ = detector.detection_pose(det, camera_params=camera_params, tag_size=float(tag_size_mm))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = pose[:3, :3]
        T[:3, 3] = pose[:3, 3]
        detections.append(
            TagDetection(
                tag_id=int(det.tag_id),
                center=tuple(np.asarray(det.center, dtype=np.float64).reshape(2).tolist()),
                corners=np.asarray(det.corners, dtype=np.float64).reshape(4, 2),
                T_camera_tag=T,
            )
        )
    return detections


def find_tag(detections, tag_id):
    for detection in detections:
        if detection.tag_id == int(tag_id):
            return detection
    return None


def draw_tags(frame_bgr, detections):
    for det in detections:
        pts = det.corners.astype(np.int32).reshape(4, 2)
        cv2.polylines(frame_bgr, [pts], True, (0, 255, 0), 2)
        c = tuple(np.asarray(det.center, dtype=np.int32).reshape(2).tolist())
        cv2.circle(frame_bgr, c, 4, (0, 0, 255), -1)
        cv2.putText(frame_bgr, f"id={det.tag_id}", (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame_bgr


def start_camera(warmup_s):
    import orbbec_native

    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(float(warmup_s))
    return cam


def detect_fixed_tag(frame, color_intrinsics, config):
    cam_cfg = config["camera"]
    detections = detect_tags(
        frame,
        color_intrinsics,
        tag_size_mm=float(cam_cfg["tag_size_mm"]),
        family=cam_cfg.get("tag_family", "tag36h11"),
    )
    fixed = find_tag(detections, cam_cfg["tag_id"])
    if fixed is None:
        return detections, None, None
    T = fixed_tag_camera_calibration(cam_cfg["T_base_tag"], fixed.T_camera_tag)
    return detections, T, fixed


def print_camera_info(camera):
    print(f"[Orbbec] color size: {camera.get_color_size()}")
    print(f"[Orbbec] depth size : {camera.get_depth_size()}")
    print(f"[Orbbec] rotate_180 : {camera.is_rotate_180_enabled()}")
    print(f"[Orbbec] color intrinsics: {camera.get_color_intrinsics()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate T_base_camera from AprilTag id=0.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--sample-delay", type=float, default=0.05)
    parser.add_argument("--max-sample-error-mm", type=float, default=20.0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    camera = start_camera(config["camera"].get("warmup_s", 1.0))
    samples = []
    misses = 0
    try:
        print_camera_info(camera)
        color_intrinsics = camera.get_color_intrinsics()
        for i in range(int(args.samples)):
            frame = camera.get_color_frame()
            if frame is None:
                misses += 1
                time.sleep(float(args.sample_delay))
                continue
            frame = np.asarray(frame, dtype=np.uint8).copy()
            detections, T, fixed = detect_fixed_tag(frame, color_intrinsics, config)
            if T is None:
                misses += 1
                print(f"[Calibration] sample {i + 1}/{args.samples}: tag id={config['camera']['tag_id']} not found")
            else:
                samples.append(T)
                t = T[:3, 3]
                tag_t = fixed.T_camera_tag[:3, 3]
                tag_dist = float(np.linalg.norm(tag_t))
                print(
                    "[Calibration] sample {}/{}: T_base_camera.t=({:.1f}, {:.1f}, {:.1f}), "
                    "tag_camera_xyz=({:.1f}, {:.1f}, {:.1f}), tag_distance={:.1f} mm".format(
                        i + 1,
                        args.samples,
                        t[0],
                        t[1],
                        t[2],
                        tag_t[0],
                        tag_t[1],
                        tag_t[2],
                        tag_dist,
                    )
                )
            if args.show:
                draw_tags(frame, detections)
                cv2.imshow("AprilTag calibration", frame)
                cv2.waitKey(1)
            time.sleep(float(args.sample_delay))

        if len(samples) < int(args.min_samples):
            raise RuntimeError(f"Only {len(samples)} valid samples; need at least {args.min_samples}")

        kept, mean_t, std_t = filter_transform_samples(samples, args.max_sample_error_mm)
        T_final = average_transforms(kept)
        print(
            "[Calibration] valid={}, kept={}, missed={}, mean=({:.2f}, {:.2f}, {:.2f}), std=({:.2f}, {:.2f}, {:.2f})".format(
                len(samples),
                len(kept),
                misses,
                mean_t[0],
                mean_t[1],
                mean_t[2],
                std_t[0],
                std_t[1],
                std_t[2],
            )
        )
        print("[Calibration] T_base_camera:")
        print(np.array2string(T_final, precision=3, suppress_small=True))
        if args.save:
            config["camera"]["T_base_camera"] = T_final.tolist()
            save_config(config, args.config)
            print("[Calibration] saved to params.json")
        else:
            print("[Calibration] dry-run only; pass --save to write params.json")
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
