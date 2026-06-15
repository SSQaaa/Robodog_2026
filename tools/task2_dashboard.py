# -*- coding: utf-8 -*-
import time

from tools.vision import analyze_infer_output


def pick_best_by_class(detections, class_id):
    best_det = None
    best_score = -1.0
    for det in detections:
        cid = int(det.get("class_id", -1))
        if cid != class_id:
            continue
        score = float(det.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_det = det
    return best_det


def box_center_x(det):
    x1, _, x2, _ = det["xyxy"]
    return (float(x1) + float(x2)) / 2.0


def pick_best_letter_detection(infer_output):
    detections = infer_output.get("detections", [])
    best_det = None
    best_score = -1.0
    for det in detections:
        cid = int(det.get("class_id", -1))
        if cid < 0 or cid > 3:
            continue
        score = float(det.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_det = det
    if best_det is None:
        return None
    x1, _, x2, _ = best_det["xyxy"]
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    cid = int(best_det["class_id"])
    return {
        "letter": letter_map.get(cid, "unknown"),
        "x_center": (float(x1) + float(x2)) / 2.0,
        "distance_m": best_det.get("distance_m", None),
        "score": float(best_det.get("score", 0.0)),
    }


def has_ssi_detection(infer_output):
    for det in infer_output.get("detections", []):
        if int(det.get("class_id", -1)) == 9:
            return True
    return False


def read_state_normal_loop(detector, need_frames=3, max_frames=40, interval_s=0.5):
    last_state = None
    same_count = 0
    frame_count = 0
    while True:
        frame_count += 1
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)
        if result["dashboard_count"] != 1:
            print("[Task2][READ_STATE] not single dashboard")
            continue
        state_cn = result["dashboard_details"][0]["state_cn"]
        if state_cn == "鏈煡":
            same_count = 0
            last_state = None
            print("[Task2][READ_STATE] unknown")
        else:
            if state_cn == last_state:
                same_count += 1
            else:
                last_state = state_cn
                same_count = 1
            print(f"[Task2][READ_STATE] current={state_cn} stable={same_count}/{need_frames}")
            if same_count >= need_frames:
                return state_cn
        if frame_count >= max_frames:
            print("[Task2][READ_STATE] timeout, return unknown")
            return "鏈煡"
        time.sleep(interval_s)


def read_letter_normal_loop(detector, need_frames=3, max_frames=40):
    last_letter = None
    same_count = 0
    frame_count = 0
    while True:
        frame_count += 1
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)
        letter = result["letter"]
        if letter == "unknown":
            same_count = 0
            last_letter = None
            print("[Task2][READ_LETTER] unknown")
        else:
            if letter == last_letter:
                same_count += 1
            else:
                last_letter = letter
                same_count = 1
            print(f"[Task2][READ_LETTER] current={letter} stable={same_count}/{need_frames}")
            if same_count >= need_frames:
                return letter
        if frame_count >= max_frames:
            print("[Task2][READ_LETTER] timeout, return unknown")
            return "unknown"
