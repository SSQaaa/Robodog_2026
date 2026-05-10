# -*- coding: utf-8 -*-
"""AprilTag detection helpers.

Preferred backend order:
1. pupil_apriltags
2. apriltag
3. cv2.aruco, only if it is available
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TagDetection:
    tag_id: int
    center: tuple
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    T_camera_tag: np.ndarray


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
        tag_id=int(tag_id),
        center=tuple(np.asarray(center, dtype=np.float64).reshape(2).tolist()),
        corners=np.asarray(corners, dtype=np.float64).reshape(4, 2),
        rvec=rvec.reshape(3, 1),
        tvec=tvec,
        T_camera_tag=T,
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
        size_mm = float(marker_size_mm)
        if marker_sizes_by_id and int(det.tag_id) in marker_sizes_by_id:
            size_mm = float(marker_sizes_by_id[int(det.tag_id)])
            # Re-run just this tag is not supported directly, so require equal
            # sizes for best precision. If sizes differ, translation scales
            # linearly with tag size.
            scale = size_mm / float(marker_size_mm)
            pose_t = np.asarray(det.pose_t, dtype=np.float64).reshape(3, 1) * scale
        else:
            pose_t = np.asarray(det.pose_t, dtype=np.float64).reshape(3, 1)

        detections.append(
            _pose_to_detection(
                det.tag_id,
                det.center,
                det.corners,
                det.pose_R,
                pose_t,
            )
        )
    return detections


def _detect_with_apriltag(gray, intrinsics, dictionary_name, marker_size_mm, marker_sizes_by_id):
    import apriltag

    family = _family_from_dictionary_name(dictionary_name)
    options = apriltag.DetectorOptions(families=family)
    detector = apriltag.Detector(options)
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

        pose, _, _ = detector.detection_pose(
            det,
            camera_params=camera_params,
            tag_size=size_mm,
        )
        detections.append(
            _pose_to_detection(
                det.tag_id,
                det.center,
                det.corners,
                pose[:3, :3],
                pose[:3, 3],
            )
        )
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
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[i]], size_mm, camera_matrix, dist_coeffs)
        R, _ = cv2.Rodrigues(rvecs[0].reshape(3, 1))
        pts = corners[i].reshape(4, 2)
        detections.append(
            _pose_to_detection(
                int(tag_id),
                np.mean(pts, axis=0),
                pts,
                R,
                tvecs[0].reshape(3, 1),
            )
        )
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


def draw_tags(frame_bgr, detections, intrinsics=None):
    if not all(hasattr(cv2, name) for name in ("polylines", "circle", "putText")):
        return frame_bgr

    for det in detections:
        pts = np.asarray(det.corners, dtype=np.int32).reshape(4, 2)
        cv2.polylines(frame_bgr, [pts], True, (0, 255, 0), 2)
        c = tuple(np.asarray(det.center, dtype=np.int32).reshape(2).tolist())
        cv2.circle(frame_bgr, c, 4, (0, 0, 255), -1)
        cv2.putText(
            frame_bgr,
            f"id={det.tag_id}",
            (c[0] + 6, c[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame_bgr
