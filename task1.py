#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4m × 1.7m 锥桶避障穿越任务（动态平移 + 目标锁定 + 边界保护）
前提：已用手柄让机器狗站立，狗位于宽度中间（0.85m处）。
逻辑：直行 → 遇锥桶 → 锁定该锥桶 → 持续同方向平移直到锥桶安全 → 恢复直行。
     若横向偏移超过0.6m，自动向中心平移修正。
"""

import os
import sys
import time
import ctypes
import threading
import socket
import struct
from collections import defaultdict, deque

import cv2
import numpy as np
import orbbec_native

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LIB_DIR = os.path.join(BASE_DIR, "/home/ysc/Desktop/2026Project/libs/")
ENGINE_PATH = os.path.join(TRT_LIB_DIR, "bigdog_0427.engine")

sys.path.append(TRT_LIB_DIR)
ctypes.CDLL(os.path.join(TRT_LIB_DIR, "libmyplugins.so"))
import yolov5_trt_cpp

# =========================
# 参数区
# =========================
# --- 检测 ---
CONF_THRESH = 0.4
MIN_VALID_DEPTH_COUNT = 20
DEPTH_HISTORY_LEN = 5

# --- 场地与机器狗 ---
TARGET_DISTANCE = 4.0       # 前进总距离（米）
FIELD_WIDTH = 1.7           # 场地宽（米）
DOG_WIDTH = 0.40
SAFE_DISTANCE = 1.5         # 开始避障的前向距离（米）
LATERAL_THRESHOLD = 0.35    # 横向判定正前方的阈值（米）
CRITICAL_DISTANCE = 0.6     # 紧急后退距离

# --- 运动控制速度值 ---
VX_NOMINAL = 8000          # 正常直行
VY_SHIFT = 25000            # 平移速度基值（正值右移）
SHIFT_DURATION = 0.4        # 单次平移基础持续时间（秒），实际会循环检查

# --- 真实速度标定（必填：你的实测速度）---
REAL_SPEED_VX = 0.5         # 米/秒，vx=12000 对应速度
LATERAL_SPEED_SCALE = 0.5 / 30000.0  # 每个 vy 单位对应的真实横向速度 (m/s)

# --- 边界保护 ---
MAX_LATERAL_OFFSET = 0.6    # 最大允许横向偏移（米）
BOUNDARY_CORRECTION_DURATION = 0.4   # 强制修正平移时间

# --- 控制周期 ---
CONTROL_PERIOD = 0.1        # 每次运动指令持续时间（秒）

# --- 窗口 ---
WINDOW_NAME = "4m Task Avoidance"

# --- 类别名 ---
CLASS_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "Green",
    5: "MPa", 6: "Red", 7: "Traffic_cone",
    8: "dashboard", 9: "ssi"
}

# =========================
# 相机内参（按实际修改）
# =========================
DEPTH_FX = 570.34
DEPTH_FY = 570.34
DEPTH_CX = 320.0
DEPTH_CY = 240.0

# =========================
# UDP 通信（同前）
# =========================
class UDPClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.send_addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)

    def __del__(self):
        self.sock.close()

    def send(self, code, value=0, type=0, last_time=0, duration=0):
        data = struct.pack("<3i", code, value, type)
        start_time = time.time()
        if last_time == 0:
            self.sock.sendto(data, self.send_addr)
            time.sleep(0.05)
        else:
            while time.time() - start_time < last_time:
                self.sock.sendto(data, self.send_addr)
                time.sleep(0.05)
        if duration != 0:
            time.sleep(duration)

class RobotMover:
    def __init__(self, ip='192.168.1.120', port=43893):
        self.udp = UDPClient(ip, port)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat(self):
        while True:
            self.udp.send(0x21040001, duration=0.2)

    def move(self, vx=0, vy=0, vz=0, last_time=0.5):
        self.udp.send(0x21010D06)          # 移动模式
        self.udp.send(0x21010300)          # 低步态
        start = time.time()
        while time.time() - start < last_time:
            self.udp.send(0x21010130, vx)
            self.udp.send(0x21010131, vy)
            self.udp.send(0x21010135, vz)
            time.sleep(0.05)

    def stop(self):
        self.udp.send(0x21010C0A, value=7)

    def __del__(self):
        self.stop()

# =========================
# 工具函数
# =========================
class DepthSmoother:
    def __init__(self, max_len=5):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, obj_id, depth_mm):
        if depth_mm <= 0:
            return None
        self.history[obj_id].append(depth_mm)
        return int(np.median(np.array(self.history[obj_id])))

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

def yolo_to_original(box, img_w, img_h, input_size=640):
    cx, cy, w, h = box
    scale = min(input_size / img_w, input_size / img_h)
    new_w = img_w * scale
    new_h = img_h * scale
    pad_x = (input_size - new_w) / 2
    pad_y = (input_size - new_h) / 2
    cx = (cx - pad_x) / scale
    cy = (cy - pad_y) / scale
    w = w / scale
    h = h / scale
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

def pixel_to_camera_3d(u, v, depth_mm, fx, fy, cx, cy):
    z = depth_mm / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def draw_detection(frame, box, cls_id, conf, depth_mm, valid_count, pos_3d=None):
    x1, y1, x2, y2 = box
    cls_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")
    if depth_mm is not None and depth_mm > 0:
        text = f"{cls_name} {conf:.2f} {depth_mm / 1000.0:.2f}m"
    else:
        text = f"{cls_name} {conf:.2f} Depth Invalid"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, text, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"ValidPts: {valid_count}",
                (x1, min(frame.shape[0] - 5, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if pos_3d is not None:
        x_3d, y_3d, z_3d = pos_3d
        coord_text = f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}m"
        cv2.putText(frame, coord_text,
                    (x1, min(frame.shape[0] - 5, y2 + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def get_avoid_direction(x, z, safe_dist=SAFE_DISTANCE, lat_thresh=LATERAL_THRESHOLD):
    """返回 0（无需避障）、1（向右平移）、-1（向左平移）"""
    if z > safe_dist or abs(x) >= lat_thresh:
        return 0
    return 1 if x < 0 else -1   # x<0 障碍在左，需向右平移 (vy > 0)

# =========================
# 主程序
# =========================
def main():
    print("[TRT] loading engine...")
    detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
    print("[TRT] engine loaded")

    print("[Robot] initializing mover...")
    mover = RobotMover()
    time.sleep(1)
    mover.stop()
    print("[Robot] ready (请确认狗已用手柄站立，且位于宽度中间)")

    print("[Orbbec] starting camera...")
    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)

    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()
    print(f"[Orbbec] color {color_w}x{color_h} depth {depth_w}x{depth_h}")

    smoother = DepthSmoother(max_len=DEPTH_HISTORY_LEN)
    forward_distance = 0.0
    lateral_offset = 0.0
    locked_cone_id = None
    lock_action = 0             # 1: right, -1: left
    shift_active = False
    shift_end_time = 0.0
    frame_id = 0

    try:
        while forward_distance < TARGET_DISTANCE:
            frame = cam.get_color_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = np.asarray(frame, dtype=np.uint8).copy()
            color_h, color_w = frame.shape[:2]

            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            print(f"[YOLO] detect num={len(detections)}, infer={(t1 - t0)*1000:.2f} ms")

            cones_3d = []
            for det in detections:
                cx_y, cy_y, w_y, h_y, conf, cls_id = det
                if conf < CONF_THRESH:
                    continue
                color_box = yolo_to_original((cx_y, cy_y, w_y, h_y),
                                             img_w=color_w, img_h=color_h)
                depth_box = scale_box(color_box, (color_w, color_h), (depth_w, depth_h))
                raw_depth, valid_count = cam.get_depth_in_box(*depth_box)

                stable_depth = None
                pos_3d = None
                if raw_depth > 0 and valid_count >= MIN_VALID_DEPTH_COUNT:
                    stable_depth = smoother.update(int(cls_id), raw_depth)
                    if stable_depth is not None:
                        u_cam = (depth_box[0] + depth_box[2]) // 2
                        v_cam = depth_box[3]
                        x_m, y_m, z_m = pixel_to_camera_3d(u_cam, v_cam, stable_depth,
                                                           DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                        pos_3d = (x_m, y_m, z_m)
                        if cls_id == 7:
                            cone_id = f"{x_m:.3f}_{z_m:.3f}_{y_m:.3f}"
                            cones_3d.append({'x': x_m, 'y': y_m, 'z': z_m,
                                             'depth': stable_depth, 'id': cone_id})
                draw_detection(frame, color_box, cls_id, conf, stable_depth, valid_count, pos_3d)

            # ========== 运动决策（目标锁定版） ==========
            if shift_active:
                if time.time() < shift_end_time:
                    mover.move(0, saved_vy, 0, last_time=CONTROL_PERIOD)
                    lateral_offset += (saved_vy * LATERAL_SPEED_SCALE) * CONTROL_PERIOD
                    print(f"[Move] shifting, lateral_offset={lateral_offset:.3f}m")
                    continue
                else:
                    shift_active = False
                    locked_cone_id = None
                    lock_action = 0

            # 边界保护（最高优先级）
            if abs(lateral_offset) > MAX_LATERAL_OFFSET:
                vy = -VY_SHIFT if lateral_offset > 0 else VY_SHIFT
                shift_active = True
                shift_end_time = time.time() + BOUNDARY_CORRECTION_DURATION
                saved_vy = vy
                locked_cone_id = None
                lock_action = 0
                mover.move(0, vy, 0, last_time=CONTROL_PERIOD)
                print(f"[Boundary] correcting, lateral_offset={lateral_offset:.2f}")
                continue

            # 正常避障逻辑
            if cones_3d:
                # 如果有锁定目标，检查它是否依然需要避让
                if locked_cone_id is not None:
                    locked_cone = next((c for c in cones_3d if c['id'] == locked_cone_id), None)
                    if locked_cone and get_avoid_direction(locked_cone['x'], locked_cone['z'], SAFE_DISTANCE, LATERAL_THRESHOLD) != 0:
                        # 继续锁定，同方向平移
                        shift_active = True
                        shift_end_time = time.time() + SHIFT_DURATION
                        saved_vy = VY_SHIFT if lock_action == 1 else -VY_SHIFT
                        mover.move(0, saved_vy, 0, last_time=CONTROL_PERIOD)
                        lateral_offset += (saved_vy * LATERAL_SPEED_SCALE) * CONTROL_PERIOD
                        print(f"[Avoid] keep avoiding locked cone x={locked_cone['x']:.3f} z={locked_cone['z']:.3f}")
                        continue
                    else:
                        # 锁定目标已安全，解除
                        locked_cone_id = None
                        lock_action = 0

                # 无锁定，选择最近的需要避障的锥桶
                nearest = min(cones_3d, key=lambda c: c['z'])
                direction = get_avoid_direction(nearest['x'], nearest['z'], SAFE_DISTANCE, LATERAL_THRESHOLD)
                if direction != 0:
                    locked_cone_id = nearest['id']
                    lock_action = direction
                    shift_active = True
                    shift_end_time = time.time() + SHIFT_DURATION
                    saved_vy = VY_SHIFT if direction == 1 else -VY_SHIFT
                    mover.move(0, saved_vy, 0, last_time=CONTROL_PERIOD)
                    lateral_offset += (saved_vy * LATERAL_SPEED_SCALE) * CONTROL_PERIOD
                    print(f"[Avoid] start avoiding cone x={nearest['x']:.3f} z={nearest['z']:.3f}, direction={'right' if direction>0 else 'left'}")
                    continue
                else:
                    # 锥桶已偏出或太远，直行
                    vx = VX_NOMINAL
                    action = "forward_cone_safe"
            else:
                vx = VX_NOMINAL
                action = "forward_no_cone"

            # 直行
            mover.move(vx, 0, 0, last_time=CONTROL_PERIOD)
            if vx > 0:
                forward_distance += REAL_SPEED_VX * CONTROL_PERIOD
            elif vx < 0:
                forward_distance -= REAL_SPEED_VX * CONTROL_PERIOD
            print(f"[Move] straight vx={vx}, total forward: {forward_distance:.3f}m, lateral_offset={lateral_offset:.3f}m")

            # 显示
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print("[User] quit by key")
                break
            frame_id += 1

    except KeyboardInterrupt:
        print("[User] interrupted")
    finally:
        print(f"[Exit] final forward_distance={forward_distance:.3f}m, lateral_offset={lateral_offset:.3f}m")
        print("[Exit] stopping robot and camera...")
        mover.stop()
        time.sleep(0.5)
        cam.stop()
        cv2.destroyAllWindows()
        print("[Exit] done")

if __name__ == "__main__":
    main()