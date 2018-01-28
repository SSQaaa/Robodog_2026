# -*- coding: utf-8 -*-
"""AprilTag guided grasp demo for Jetson Xavier + OrbbecCamera wrapper."""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from three_Inverse_kinematics import Arm  # noqa: E402
except ModuleNotFoundError:
    Arm = None


try:
    from camera_adapter import OrbbecCameraAdapter
    from coordinate_transform import (
        as_transform,
        base_point_to_arm_target,
        fixed_tag_camera_calibration,
        pixel_to_camera,
        matrix_to_list,
        object_tag_to_grasp_point,
    )
    from tag_detector import detect_tags, draw_tags, find_tag
except ModuleNotFoundError:
    import math

    try:
        import orbbec_native
    except ModuleNotFoundError:
        orbbec_native = None

    class OrbbecCameraAdapter:
        def __init__(self, warmup_s=1.0):
            if orbbec_native is None:
                raise RuntimeError("orbbec_native is not available. Build/install the Orbbec wrapper on the robot.")
            self.cam = orbbec_native.OrbbecCamera()
            self.warmup_s = warmup_s
            self.color_intrinsics = None
            self.depth_intrinsics = None

        def start(self):
            self.cam.start()
            time.sleep(self.warmup_s)
            self.color_intrinsics = self.cam.get_color_intrinsics()
            self.depth_intrinsics = self.cam.get_depth_intrinsics()
            return self

        def stop(self):
            self.cam.stop()

        def get_frame(self):
            frame = self.cam.get_color_frame()
            if frame is None:
                return None
            return np.asarray(frame, dtype=np.uint8).copy()

        def get_color_size(self):
            return self.cam.get_color_size()

        def get_depth_size(self):
            return self.cam.get_depth_size()

        def is_rotate_180_enabled(self):
            return self.cam.is_rotate_180_enabled()

        def color_to_depth_pixel(self, u, v):
            color_w, color_h = self.get_color_size()
            depth_w, depth_h = self.get_depth_size()
            x = int(round(float(u) * float(depth_w) / float(color_w)))
            y = int(round(float(v) * float(depth_h) / float(color_h)))
            x = max(0, min(depth_w - 1, x))
            y = max(0, min(depth_h - 1, y))
            return x, y

        def get_depth_at_color_pixel(self, u, v, box_radius=8, min_valid_count=20):
            depth_x, depth_y = self.color_to_depth_pixel(u, v)
            depth_mm, valid_count = self.cam.get_depth_in_box(
                depth_x - box_radius,
                depth_y - box_radius,
                depth_x + box_radius,
                depth_y + box_radius,
            )
            if depth_mm <= 0 or valid_count < min_valid_count:
                return None, valid_count
            return int(depth_mm), int(valid_count)

    class TagDetection:
        def __init__(self, tag_id, center, corners, rvec, tvec, T_camera_tag):
            self.tag_id = tag_id
            self.center = center
            self.corners = corners
            self.rvec = rvec
            self.tvec = tvec
            self.T_camera_tag = T_camera_tag

    def as_transform(value):
        T = np.asarray(value, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        return T

    def transform_point(T, point):
        p = np.asarray(point, dtype=np.float64).reshape(3)
        out = as_transform(T) @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        return out[:3]

    def object_tag_to_grasp_point(T_base_tag, grasp_offset_tag_mm):
        return transform_point(T_base_tag, grasp_offset_tag_mm)

    def pixel_to_camera(u, v, depth_mm, intrinsics):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x = (float(u) - cx) * float(depth_mm) / fx
        y = (float(v) - cy) * float(depth_mm) / fy
        z = float(depth_mm)
        return np.array([x, y, z], dtype=np.float64)

    def fixed_tag_camera_calibration(T_base_fixed_tag, T_camera_fixed_tag):
        return as_transform(T_base_fixed_tag) @ np.linalg.inv(as_transform(T_camera_fixed_tag))

    def matrix_to_list(T):
        return np.asarray(T, dtype=float).tolist()

    def base_point_to_arm_target(point_base, gripper_offset_mm, shoulder_height_mm):
        x, y, z = np.asarray(point_base, dtype=np.float64).reshape(3)
        yaw_rad = math.atan2(y, x)
        reach_mm = math.sqrt(x * x + y * y) - float(gripper_offset_mm)
        height_mm = z - float(shoulder_height_mm)
        return {
            "yaw_rad": yaw_rad,
            "yaw_deg": math.degrees(yaw_rad),
            "reach_mm": reach_mm,
            "height_mm": height_mm,
        }

    def intrinsics_to_camera_matrix(intrinsics):
        return np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _bgr_to_gray(frame_bgr):
        if hasattr(cv2, "cvtColor") and hasattr(cv2, "COLOR_BGR2GRAY"):
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        frame = np.asarray(frame_bgr)
        if frame.ndim == 2:
            return frame.astype(np.uint8, copy=False)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError("Expected a gray or BGR image for AprilTag detection")

        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _rotation_matrix_to_rvec(R):
        if hasattr(cv2, "Rodrigues"):
            rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
            return rvec.reshape(3, 1)

        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = np.trace(R)
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if abs(theta) < 1e-9:
            return np.zeros((3, 1), dtype=np.float64)

        denom = 2.0 * np.sin(theta)
        axis = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=np.float64,
        ) / denom
        return (axis * theta).reshape(3, 1)

    def _family_from_dictionary_name(dictionary_name):
        name = str(dictionary_name).lower()
        if "36h11" in name:
            return "tag36h11"
        if "25h9" in name:
            return "tag25h9"
        if "16h5" in name:
            return "tag16h5"
        return "tag36h11"

    def _pose_to_detection(tag_id, center, corners, R, t):
        rvec = _rotation_matrix_to_rvec(R)
        tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
        T[:3, 3] = tvec.reshape(3)
        return TagDetection(
            int(tag_id),
            tuple(np.asarray(center, dtype=np.float64).reshape(2).tolist()),
            np.asarray(corners, dtype=np.float64).reshape(4, 2),
            rvec.reshape(3, 1),
            tvec,
            T,
        )

    def _detect_with_pupil_apriltags(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id):
        from pupil_apriltags import Detector

        detector = Detector(
            families=_family_from_dictionary_name(dictionary_name),
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        camera_params = [
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        ]
        raw = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=float(marker_size_mm),
        )
        detections = []
        for det in raw:
            pose_t = np.asarray(det.pose_t, dtype=np.float64).reshape(3, 1)
            if marker_sizes_by_id and int(det.tag_id) in marker_sizes_by_id:
                pose_t = pose_t * (float(marker_sizes_by_id[int(det.tag_id)]) / float(marker_size_mm))
            detections.append(_pose_to_detection(det.tag_id, det.center, det.corners, det.pose_R, pose_t))
        return detections

    def _detect_with_apriltag(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id):
        import apriltag

        detector = apriltag.Detector(
            apriltag.DetectorOptions(families=_family_from_dictionary_name(dictionary_name))
        )
        raw = detector.detect(gray)
        camera_params = (
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        )
        detections = []
        for det in raw:
            size_mm = float(marker_size_mm)
            if marker_sizes_by_id and int(det.tag_id) in marker_sizes_by_id:
                size_mm = float(marker_sizes_by_id[int(det.tag_id)])
            pose, _, _ = detector.detection_pose(det, camera_params=camera_params, tag_size=size_mm)
            detections.append(_pose_to_detection(det.tag_id, det.center, det.corners, pose[:3, :3], pose[:3, 3]))
        return detections

    def _detect_with_cv2_aruco(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id):
        if not hasattr(cv2, "aruco"):
            raise ImportError("cv2.aruco is not available")

        aruco = cv2.aruco
        if not hasattr(aruco, dictionary_name):
            raise RuntimeError(f"OpenCV aruco dictionary not available: {dictionary_name}")
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
        params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()

        if hasattr(aruco, "ArucoDetector"):
            corners, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
        if ids is None or len(ids) == 0:
            return []

        camera_matrix = intrinsics_to_camera_matrix(intrinsics)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        detections = []

        for i, tag_id in enumerate(ids.flatten()):
            size_mm = float(marker_size_mm)
            if marker_sizes_by_id and int(tag_id) in marker_sizes_by_id:
                size_mm = float(marker_sizes_by_id[int(tag_id)])
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                [corners[i]],
                size_mm,
                camera_matrix,
                dist_coeffs,
            )
            pts = corners[i].reshape(4, 2)
            R, _ = cv2.Rodrigues(rvecs[0].reshape(3, 1))
            detections.append(_pose_to_detection(int(tag_id), np.mean(pts, axis=0), pts, R, tvecs[0]))
        return detections

    def detect_tags(
        frame_bgr,
        intrinsics,
        marker_size_mm,
        dictionary_name="DICT_APRILTAG_36h11",
        marker_sizes_by_id=None,
        backend="auto",
    ):
        gray = _bgr_to_gray(frame_bgr)
        backends = [backend] if backend != "auto" else ["pupil_apriltags", "apriltag", "cv2_aruco"]
        errors = []
        for name in backends:
            try:
                if name == "pupil_apriltags":
                    return _detect_with_pupil_apriltags(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id)
                if name == "apriltag":
                    return _detect_with_apriltag(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id)
                if name == "cv2_aruco":
                    return _detect_with_cv2_aruco(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id)
                raise RuntimeError(f"Unknown tag backend: {name}")
            except (ImportError, ModuleNotFoundError, AttributeError) as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError("No AprilTag backend is available. Tried: " + "; ".join(errors))

    def find_tag(detections, tag_id):
        for det in detections:
            if det.tag_id == int(tag_id):
                return det
        return None

    def draw_tags(frame_bgr, detections, intrinsics):
        if not all(hasattr(cv2, name) for name in ("polylines", "circle", "putText")):
            return frame_bgr

        for det in detections:
            pts = np.asarray(det.corners, dtype=np.int32).reshape(4, 2)
            cv2.polylines(frame_bgr, [pts], True, (0, 255, 0), 2)
            c = tuple(np.asarray(det.center, dtype=np.int32).reshape(2).tolist())
            cv2.circle(frame_bgr, c, 4, (0, 0, 255), -1)
            cv2.putText(frame_bgr, f"id={det.tag_id}", (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame_bgr


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_to_base_servo(yaw_deg, base_cfg):
    raw = (
        float(base_cfg["center"])
        + float(base_cfg.get("direction", 1)) * float(yaw_deg) * float(base_cfg["ticks_per_degree"])
    )
    return int(round(clamp(raw, int(base_cfg["min"]), int(base_cfg["max"]))))


def transform_point_local(T, point):
    p = np.asarray(point, dtype=np.float64).reshape(3)
    hp = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (as_transform(T) @ hp)[:3]


def camera_to_base_point(point_camera, T_base_camera):
    return transform_point_local(T_base_camera, point_camera)


def average_transforms(transforms):
    mats = np.asarray([as_transform(T) for T in transforms], dtype=np.float64)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError("Expected a non-empty list of 4x4 transforms")

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
        return [], np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    mats = np.asarray([as_transform(T) for T in transforms], dtype=np.float64)
    translations = mats[:, :3, 3]
    median_t = np.median(translations, axis=0)
    errors = np.linalg.norm(translations - median_t, axis=1)
    kept = [
        transforms[i]
        for i, err in enumerate(errors)
        if err <= float(max_translation_error_mm)
    ]
    if not kept:
        kept = list(transforms)
    kept_t = np.asarray([as_transform(T)[:3, 3] for T in kept], dtype=np.float64)
    return kept, np.mean(kept_t, axis=0), np.std(kept_t, axis=0)


def hsv_bounds(block_cfg):
    hsv_cfg = block_cfg.get("hsv", {})
    lower = np.asarray(hsv_cfg.get("lower", [35, 40, 40]), dtype=np.uint8)
    upper = np.asarray(hsv_cfg.get("upper", [85, 255, 255]), dtype=np.uint8)
    return lower, upper


def detect_green_block(frame_bgr, config):
    block_cfg = config.get("block", {})
    lower, upper = hsv_bounds(block_cfg)
    min_area = float(block_cfg.get("min_area_px", 500.0))
    kernel_size = int(block_cfg.get("morph_kernel_px", 5))
    kernel_size = max(1, kernel_size)

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
    area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(contour)

    return {
        "center": (float(cx), float(cy)),
        "area": area,
        "angle_deg": float(angle),
        "rect_size": (float(w), float(h)),
        "box": box,
        "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
    }, mask


def draw_block_detection(frame_bgr, detection, result=None):
    if detection is None:
        return frame_bgr
    cv2.drawContours(frame_bgr, [detection["box"]], 0, (0, 255, 0), 2)
    cx, cy = detection["center"]
    cv2.circle(frame_bgr, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
    label = "green block"
    if result is not None and result.get("depth_mm") is not None:
        label = f"{label} {result['depth_mm'] / 1000.0:.2f}m"
    x1, y1, _, _ = detection["bbox"]
    cv2.putText(
        frame_bgr,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return frame_bgr


def check_joint_limits(servo_targets, limits):
    for servo_id, value in servo_targets.items():
        key = str(servo_id)
        if key not in limits:
            continue
        low, high = limits[key]
        if not (int(low) <= int(value) <= int(high)):
            raise ValueError(f"Servo {servo_id} target {value} is outside limit [{low}, {high}]")


def limit_margin(servo_targets, limits):
    margins = []
    for servo_id, value in servo_targets.items():
        key = str(servo_id)
        if key not in limits:
            continue
        low, high = limits[key]
        margins.append(min(int(value) - int(low), int(high) - int(value)))
    if not margins:
        return 0
    return min(margins)


def solve_arm_with_theta(arm_target, arm_cfg):
    if Arm is None:
        raise RuntimeError("three_Inverse_kinematics.py is not available next to grasp_demo_orbbec.py")

    theta_candidates = arm_cfg.get("theta_candidates_deg", [0.0])
    limits = arm_cfg["joint_limits"]
    ids = arm_cfg["ids"]
    valid = []
    errors = []

    for theta_deg in theta_candidates:
        try:
            angle_3, angle_4, angle_5 = Arm(
                arm_target["reach_mm"],
                arm_target["height_mm"],
                theta_deg=float(theta_deg),
            )
            link_targets = {
                ids["link3"]: angle_3,
                ids["link4"]: angle_4,
                ids["link5"]: angle_5,
            }
            margin = limit_margin(link_targets, limits)
            print(
                "[ThetaTry] theta={:.1f}, targets={}, margin={}".format(
                    float(theta_deg),
                    link_targets,
                    margin,
                )
            )
            check_joint_limits(link_targets, limits)
            valid.append((margin, float(theta_deg), angle_3, angle_4, angle_5))
        except Exception as exc:
            errors.append((float(theta_deg), str(exc)))
            print(f"[ThetaTry] theta={float(theta_deg):.1f} rejected: {exc}")

    if not valid:
        raise ValueError(f"No valid IK theta candidate. Errors: {errors}")

    valid.sort(key=lambda item: item[0], reverse=True)
    margin, theta_deg, angle_3, angle_4, angle_5 = valid[0]
    print(f"[ThetaPick] theta={theta_deg:.1f}, margin={margin}")
    return angle_3, angle_4, angle_5, theta_deg


class RobotArmController:
    def __init__(self, arm_cfg):
        from scservo_sdk import PortHandler, sms_sts  # noqa: WPS433

        self.cfg = arm_cfg
        self.ids = arm_cfg["ids"]
        self.speed = int(arm_cfg["moving_speed"])
        self.acc = int(arm_cfg["moving_acc"])
        self.port = PortHandler(arm_cfg["devicename"])
        self.packet = sms_sts(self.port)

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open servo port: {arm_cfg['devicename']}")
        if not self.port.setBaudRate(int(arm_cfg["baudrate"])):
            raise RuntimeError(f"Failed to set servo baudrate: {arm_cfg['baudrate']}")
        print("[Arm] servo port opened")

    def close(self):
        self.port.closePort()

    def write_servo(self, servo_id, position):
        result, error = self.packet.WritePosEx(int(servo_id), int(position), self.speed, self.acc)
        if result != 0:
            print(self.packet.getTxRxResult(result))
        if error != 0:
            print(self.packet.getRxPacketError(error))

    def move_targets(self, targets, wait_s=1.2):
        for servo_id, position in targets.items():
            self.write_servo(servo_id, position)
        time.sleep(wait_s)

    def gripper_open(self):
        self.write_servo(self.ids["gripper"], self.cfg["gripper"]["open"])
        time.sleep(0.8)

    def gripper_close(self):
        self.write_servo(self.ids["gripper"], self.cfg["gripper"]["close"])
        time.sleep(0.8)

    def move_to_arm_target(self, arm_target, wait_s=1.5):
        print(
            "[ArmDebug] yaw={:.1f} deg, reach={:.1f} mm, height={:.1f} mm".format(
                arm_target["yaw_deg"],
                arm_target["reach_mm"],
                arm_target["height_mm"],
            )
        )
        angle_3, angle_4, angle_5, theta_deg = solve_arm_with_theta(arm_target, self.cfg)
        targets = {
            self.ids["base"]: yaw_to_base_servo(arm_target["yaw_deg"], self.cfg["base_servo"]),
            self.ids["link3"]: angle_3,
            self.ids["link4"]: angle_4,
            self.ids["link5"]: angle_5,
        }
        self.move_targets(targets, wait_s=wait_s)
        return targets


def wait_for_frame(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.02)


def detect_once(camera, config):
    frame = wait_for_frame(camera)
    intrinsics = camera.color_intrinsics
    marker_size = float(config["object_tag"]["size_mm"])
    marker_sizes = {
        int(config["fixed_tag"]["id"]): float(config["fixed_tag"]["size_mm"]),
        int(config["object_tag"]["id"]): float(config["object_tag"]["size_mm"]),
    }
    detections = detect_tags(
        frame,
        intrinsics,
        marker_size_mm=marker_size,
        dictionary_name=config.get("aruco_dictionary", "DICT_APRILTAG_36h11"),
        marker_sizes_by_id=marker_sizes,
        backend=config.get("tag_backend", "auto"),
    )
    return frame, detections


def calibrate_from_fixed_tag(config, detections):
    fixed_cfg = config["fixed_tag"]
    fixed = find_tag(detections, fixed_cfg["id"])
    if fixed is None:
        return None
    T_base_fixed_tag = as_transform(fixed_cfg["T_base_tag"])
    return fixed_tag_camera_calibration(T_base_fixed_tag, fixed.T_camera_tag)


def resolve_T_base_camera(config, detections, allow_saved=True):
    T = calibrate_from_fixed_tag(config, detections)
    if T is not None:
        return T, "fixed-tag"
    if allow_saved and config.get("T_base_camera") is not None:
        return as_transform(config["T_base_camera"]), "saved"
    raise RuntimeError("No fixed tag detected and no saved T_base_camera in calibration.json")


def compute_grasp(config, detections, T_base_camera):
    object_cfg = config["object_tag"]
    obj = find_tag(detections, object_cfg["id"])
    if obj is None:
        raise RuntimeError(f"Object tag id {object_cfg['id']} not detected")

    T_base_object_tag = as_transform(T_base_camera) @ obj.T_camera_tag
    grasp_point_base = object_tag_to_grasp_point(
        T_base_object_tag,
        object_cfg["grasp_offset_tag_mm"],
    )
    grasp_offset_base = np.asarray(object_cfg.get("grasp_offset_base_mm", [0.0, 0.0, 0.0]), dtype=np.float64)
    grasp_point_base = grasp_point_base + grasp_offset_base
    arm_target = base_point_to_arm_target(
        grasp_point_base,
        gripper_offset_mm=config["arm"]["gripper_offset_mm"],
        shoulder_height_mm=config["arm"]["shoulder_height_mm"],
    )

    print(
        "[ArmDebug] grasp_point_base=({:.1f}, {:.1f}, {:.1f}) mm".format(
            grasp_point_base[0],
            grasp_point_base[1],
            grasp_point_base[2],
        )
    )
    print(
        "[ArmDebug] yaw={:.1f} deg, reach={:.1f} mm, height={:.1f} mm".format(
            arm_target["yaw_deg"],
            arm_target["reach_mm"],
            arm_target["height_mm"],
        )
    )

    angle_3, angle_4, angle_5, theta_deg = solve_arm_with_theta(arm_target, config["arm"])
    servo_targets = {
        config["arm"]["ids"]["base"]: yaw_to_base_servo(arm_target["yaw_deg"], config["arm"]["base_servo"]),
        config["arm"]["ids"]["link3"]: angle_3,
        config["arm"]["ids"]["link4"]: angle_4,
        config["arm"]["ids"]["link5"]: angle_5,
    }

    return {
        "object_tag_id": obj.tag_id,
        "object_tag_center": obj.center,
        "T_base_object_tag": T_base_object_tag,
        "grasp_point_base": grasp_point_base,
        "arm_target": arm_target,
        "theta_deg": theta_deg,
        "servo_targets": servo_targets,
    }


def compute_block_grasp(config, camera, frame, T_base_camera):
    block_cfg = config.get("block", {})
    detection, mask = detect_green_block(frame, config)
    if detection is None:
        raise RuntimeError("Green block not detected. Adjust block.hsv or block.min_area_px in calibration.json")

    u, v = detection["center"]
    depth_cfg = block_cfg.get("depth_roi", {})
    depth_mm, valid_count = camera.get_depth_at_color_pixel(
        u,
        v,
        box_radius=int(depth_cfg.get("box_radius_px", 8)),
        min_valid_count=int(depth_cfg.get("min_valid_count", 20)),
    )
    if depth_mm is None:
        raise RuntimeError(
            "No valid depth at green block center "
            f"(valid_count={valid_count}). Increase depth_roi.box_radius_px or improve depth view."
        )

    point_camera = pixel_to_camera(u, v, depth_mm, camera.color_intrinsics)
    point_base_from_depth = camera_to_base_point(point_camera, T_base_camera)

    size_mm = np.asarray(block_cfg.get("size_mm", [100.0, 50.0, 50.0]), dtype=np.float64)
    if size_mm.size < 3:
        raise ValueError("block.size_mm must contain [length, width, height]")
    plane_z = float(block_cfg.get("plane_z_base_mm", 0.0))
    grasp_z_offset = float(block_cfg.get("grasp_z_offset_mm", 0.0))
    grasp_point_base = point_base_from_depth.copy()
    grasp_point_base[2] = plane_z + float(size_mm[2]) * 0.5 + grasp_z_offset

    grasp_offset_base = np.asarray(block_cfg.get("grasp_offset_base_mm", [0.0, 0.0, 0.0]), dtype=np.float64)
    grasp_point_base = grasp_point_base + grasp_offset_base

    arm_target = base_point_to_arm_target(
        grasp_point_base,
        gripper_offset_mm=config["arm"]["gripper_offset_mm"],
        shoulder_height_mm=config["arm"]["shoulder_height_mm"],
    )
    angle_3, angle_4, angle_5, theta_deg = solve_arm_with_theta(arm_target, config["arm"])
    servo_targets = {
        config["arm"]["ids"]["base"]: yaw_to_base_servo(arm_target["yaw_deg"], config["arm"]["base_servo"]),
        config["arm"]["ids"]["link3"]: angle_3,
        config["arm"]["ids"]["link4"]: angle_4,
        config["arm"]["ids"]["link5"]: angle_5,
    }

    return {
        "detection": detection,
        "mask": mask,
        "pixel": (float(u), float(v)),
        "depth_mm": int(depth_mm),
        "valid_count": int(valid_count),
        "point_camera": point_camera,
        "point_base_from_depth": point_base_from_depth,
        "grasp_point_base": grasp_point_base,
        "arm_target": arm_target,
        "theta_deg": theta_deg,
        "servo_targets": servo_targets,
    }


def print_camera_info(camera):
    print(f"[Orbbec] color size: {camera.get_color_size()}")
    print(f"[Orbbec] depth size : {camera.get_depth_size()}")
    print(f"[Orbbec] rotate_180 : {camera.is_rotate_180_enabled()}")
    print(f"[Orbbec] color intrinsics: {camera.color_intrinsics}")


def print_grasp_result(result, source):
    p = result["grasp_point_base"]
    target = result["arm_target"]
    print(f"[Calibration] T_base_camera source: {source}")
    print(f"[Object] tag id: {result['object_tag_id']}, center: {result['object_tag_center']}")
    print(f"[Base] grasp XYZ: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
    print(
        "[Arm] yaw={:.1f} deg, reach={:.1f} mm, height={:.1f} mm, theta={:.1f} deg".format(
            target["yaw_deg"],
            target["reach_mm"],
            target["height_mm"],
            result["theta_deg"],
        )
    )
    print(f"[Servo] targets: {result['servo_targets']}")


def print_block_result(result, source):
    p = result["grasp_point_base"]
    raw = result["point_base_from_depth"]
    cam = result["point_camera"]
    target = result["arm_target"]
    print(f"[Calibration] T_base_camera source: {source}")
    print(
        "[Block] pixel=({:.1f},{:.1f}), depth={} mm, valid={}".format(
            result["pixel"][0],
            result["pixel"][1],
            result["depth_mm"],
            result["valid_count"],
        )
    )
    print("[Camera] point XYZ=({:.1f}, {:.1f}, {:.1f}) mm".format(cam[0], cam[1], cam[2]))
    print("[Base] depth XYZ=({:.1f}, {:.1f}, {:.1f}) mm".format(raw[0], raw[1], raw[2]))
    print("[Base] grasp XYZ=({:.1f}, {:.1f}, {:.1f}) mm".format(p[0], p[1], p[2]))
    print(
        "[Arm] yaw={:.1f} deg, reach={:.1f} mm, height={:.1f} mm, theta={:.1f} deg".format(
            target["yaw_deg"],
            target["reach_mm"],
            target["height_mm"],
            result["theta_deg"],
        )
    )
    print(f"[Servo] targets: {result['servo_targets']}")


def run_camera_test(args, config):
    camera = OrbbecCameraAdapter().start()
    try:
        print_camera_info(camera)
        frame, detections = detect_once(camera, config)
        print(f"[Tag] detected ids: {[det.tag_id for det in detections]}")
        for det in detections:
            depth_mm, valid_count = camera.get_depth_at_color_pixel(*det.center)
            print(
                f"[Tag] id={det.tag_id}, center=({det.center[0]:.1f},{det.center[1]:.1f}), "
                f"depth={depth_mm}, valid={valid_count}"
            )
        if args.show:
            draw_tags(frame, detections, camera.color_intrinsics)
            cv2.imshow("Orbbec AprilTag Test", frame)
            cv2.waitKey(0)
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


def run_calibrate(args, config):
    camera = OrbbecCameraAdapter().start()
    try:
        print_camera_info(camera)
        _, detections = detect_once(camera, config)
        T = calibrate_from_fixed_tag(config, detections)
        if T is None:
            raise RuntimeError(f"Fixed tag id {config['fixed_tag']['id']} not detected")
        config["T_base_camera"] = matrix_to_list(T)
        save_config(args.config, config)
        print("[Calibration] saved T_base_camera:")
        print(np.array2string(T, precision=3, suppress_small=True))
    finally:
        camera.stop()


def run_calibrate_samples(args, config):
    camera = OrbbecCameraAdapter().start()
    samples = []
    misses = 0
    try:
        print_camera_info(camera)
        print(
            "[Calibration] collecting up to {} samples, need at least {}".format(
                args.samples,
                args.min_samples,
            )
        )
        for i in range(int(args.samples)):
            _, detections = detect_once(camera, config)
            T = calibrate_from_fixed_tag(config, detections)
            if T is None:
                misses += 1
                print(f"[Calibration] sample {i + 1}/{args.samples}: fixed tag not detected")
            else:
                samples.append(T)
                t = T[:3, 3]
                print(
                    "[Calibration] sample {}/{}: t=({:.1f}, {:.1f}, {:.1f}) mm".format(
                        i + 1,
                        args.samples,
                        t[0],
                        t[1],
                        t[2],
                    )
                )
            time.sleep(float(args.sample_delay))

        if len(samples) < int(args.min_samples):
            raise RuntimeError(
                f"Only collected {len(samples)} valid samples; need at least {args.min_samples}. "
                "Keep the fixed tag flat, fully visible, and well lit."
            )

        kept, t_mean, t_std = filter_transform_samples(samples, args.max_sample_error_mm)
        T = average_transforms(kept)
        config["T_base_camera"] = matrix_to_list(T)
        save_config(args.config, config)

        print(
            "[Calibration] valid={}, kept={}, missed={}, translation std=({:.2f}, {:.2f}, {:.2f}) mm".format(
                len(samples),
                len(kept),
                misses,
                t_std[0],
                t_std[1],
                t_std[2],
            )
        )
        print(
            "[Calibration] translation mean=({:.2f}, {:.2f}, {:.2f}) mm".format(
                t_mean[0],
                t_mean[1],
                t_mean[2],
            )
        )
        print("[Calibration] saved averaged T_base_camera:")
        print(np.array2string(T, precision=3, suppress_small=True))
    finally:
        camera.stop()


def run_dry_or_grasp(args, config):
    camera = OrbbecCameraAdapter().start()
    arm = None
    try:
        print_camera_info(camera)
        _, detections = detect_once(camera, config)
        T_base_camera, source = resolve_T_base_camera(config, detections, allow_saved=True)
        result = compute_grasp(config, detections, T_base_camera)
        print_grasp_result(result, source)

        if not args.execute:
            print("[DryRun] not moving servos. Add --execute to move the arm.")
            return

        arm = RobotArmController(config["arm"])
        lift = float(config["arm"]["pre_grasp_lift_mm"])
        pre_point = result["grasp_point_base"].copy()
        pre_point[2] += lift
        pre_target = base_point_to_arm_target(
            pre_point,
            gripper_offset_mm=config["arm"]["gripper_offset_mm"],
            shoulder_height_mm=config["arm"]["shoulder_height_mm"],
        )

        arm.gripper_open()
        print("[Arm] moving to pre-grasp")
        arm.move_to_arm_target(pre_target, wait_s=1.8)
        print("[Arm] moving to grasp")
        arm.move_to_arm_target(result["arm_target"], wait_s=1.8)
        arm.gripper_close()
        print("[Arm] lifting")
        arm.move_to_arm_target(pre_target, wait_s=1.8)
    finally:
        if arm is not None:
            arm.close()
        camera.stop()


def run_block_test(args, config):
    camera = OrbbecCameraAdapter().start()
    try:
        print_camera_info(camera)
        frame = wait_for_frame(camera)
        detections = []
        try:
            _, detections = detect_once(camera, config)
        except RuntimeError as exc:
            print(f"[Tag] live fixed-tag detection skipped: {exc}")
        T_base_camera, source = resolve_T_base_camera(config, detections, allow_saved=True)
        result = compute_block_grasp(config, camera, frame, T_base_camera)
        print_block_result(result, source)
        if args.show:
            draw_block_detection(frame, result["detection"], result)
            cv2.imshow("Green Block Test", frame)
            cv2.imshow("Green Block Mask", result["mask"])
            cv2.waitKey(0)
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


def run_block_dry_or_grasp(args, config):
    camera = OrbbecCameraAdapter().start()
    arm = None
    try:
        print_camera_info(camera)
        frame = wait_for_frame(camera)
        detections = []
        try:
            _, detections = detect_once(camera, config)
        except RuntimeError as exc:
            print(f"[Tag] live fixed-tag detection skipped: {exc}")
        T_base_camera, source = resolve_T_base_camera(config, detections, allow_saved=True)
        result = compute_block_grasp(config, camera, frame, T_base_camera)
        print_block_result(result, source)

        if not args.execute:
            print("[DryRun] not moving servos. Add --execute to move the arm.")
            return

        arm = RobotArmController(config["arm"])
        lift = float(config["arm"]["pre_grasp_lift_mm"])
        pre_point = result["grasp_point_base"].copy()
        pre_point[2] += lift
        pre_target = base_point_to_arm_target(
            pre_point,
            gripper_offset_mm=config["arm"]["gripper_offset_mm"],
            shoulder_height_mm=config["arm"]["shoulder_height_mm"],
        )

        arm.gripper_open()
        print("[Arm] moving to pre-grasp")
        arm.move_to_arm_target(pre_target, wait_s=1.8)
        print("[Arm] moving to grasp")
        arm.move_to_arm_target(result["arm_target"], wait_s=1.8)
        arm.gripper_close()
        print("[Arm] lifting")
        arm.move_to_arm_target(pre_target, wait_s=1.8)
    finally:
        if arm is not None:
            arm.close()
        camera.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Orbbec AprilTag grasp demo")
    parser.add_argument(
        "mode",
        choices=[
            "camera-test",
            "calibrate",
            "calibrate-samples",
            "dry-run",
            "grasp",
            "block-test",
            "block-dry-run",
            "block-grasp",
        ],
        help="Run mode. grasp also requires --execute to move servos.",
    )
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "calibration.json"))
    parser.add_argument("--show", action="store_true", help="Show detected tag window in camera-test")
    parser.add_argument("--execute", action="store_true", help="Actually move servos")
    parser.add_argument("--samples", type=int, default=60, help="Frame count for calibrate-samples")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum valid fixed-tag samples")
    parser.add_argument("--sample-delay", type=float, default=0.05, help="Delay between calibration samples in seconds")
    parser.add_argument(
        "--max-sample-error-mm",
        type=float,
        default=20.0,
        help="Reject calibration samples farther than this from the median translation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "camera-test":
        run_camera_test(args, config)
    elif args.mode == "calibrate":
        run_calibrate(args, config)
    elif args.mode == "calibrate-samples":
        run_calibrate_samples(args, config)
    elif args.mode in ("dry-run", "grasp"):
        if args.mode == "grasp" and not args.execute:
            print("[Safety] grasp mode selected without --execute, running as dry-run.")
        run_dry_or_grasp(args, config)
    elif args.mode == "block-test":
        run_block_test(args, config)
    elif args.mode in ("block-dry-run", "block-grasp"):
        if args.mode == "block-grasp" and not args.execute:
            print("[Safety] block-grasp mode selected without --execute, running as dry-run.")
        run_block_dry_or_grasp(args, config)


if __name__ == "__main__":
    main()
