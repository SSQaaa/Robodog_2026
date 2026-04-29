#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2 米避障测试（平移绕过版本）
前提：已用手柄让机器狗站立，并处于可移动状态。
逻辑：直行 → 遇到锥桶 → 根据横向位置向左/右平移一段固定时间 → 继续直行 → 累计前进距离达到 2 米时停止。
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
TARGET_DISTANCE = 2.0       # 需要前进的总距离（米）
DOG_LENGTH = 0.55
DOG_WIDTH = 0.40
SAFE_DISTANCE = 1.2         # 开始避障的前向距离（米）
LATERAL_THRESHOLD = 0.35    # 横向判定正前方的阈值（米）
CRITICAL_DISTANCE = 0.5     # 太近时后退（保留，但平移动作用不到）

# --- 运动控制速度值 ---
VX_NOMINAL = 12000          # 正常直行
VX_SLOW = 8000              # 未使用（平移时不前进）
VY_SHIFT = 30000            # 平移速度（左/右）
SHIFT_DURATION = 0.6        # 平移持续时间（秒），可根据效果调整

# --- 真实速度标定（vx=12000 对应的实际速度）---
REAL_SPEED_VX = 0.8         # 米/秒，请用你的标定结果替换

# --- 控制周期 ---
CONTROL_PERIOD = 0.1        # 每次运动指令持续时间（秒）

# --- 窗口 ---
WINDOW_NAME = "2m Shift Avoidance"

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
        """基础移动，持续 last_time 秒"""
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
# 工具函数（同上）
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

# =========================
# 避障决策（平移版本）
# =========================
def compute_avoidance_action(x, z,
                             safe_dist=SAFE_DISTANCE,
                             critical_dist=CRITICAL_DISTANCE,
                             lat_thresh=LATERAL_THRESHOLD):
    """
    返回 (vx, vy, vz, description)
    逻辑：
      - 安全距离外：直行
      - 避障区内：若正前方（|x|<阈值），则横向平移；否则直行（已偏）
      - 危险距离内：后退+平移（仍用平移远离）
    """
    if z > safe_dist:
        return VX_NOMINAL, 0, 0, "forward"
    elif critical_dist < z <= safe_dist:
        if abs(x) < lat_thresh:
            # 正前方，向远离锥桶的一侧平移
            if x >= 0:
                return 0, -VY_SHIFT, 0, "shift_right"   # 锥桶在右，向右平移（vy正值向右？注意坐标定义）
            else:
                return 0, VY_SHIFT, 0, "shift_left"
        else:
            # 已经偏出，可直行
            return VX_NOMINAL, 0, 0, "forward_offcenter"
    else:  # z <= critical_dist
        # 太近，后退+平移躲避
        if x >= 0:
            return -6000, -VY_SHIFT, 0, "back_shift_right"
        else:
            return -6000, VY_SHIFT, 0, "back_shift_left"

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
    print("[Robot] ready (请确认狗已用手柄站立)")

    print("[Orbbec] starting camera...")
    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)

    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()
    print(f"[Orbbec] color {color_w}x{color_h} depth {depth_w}x{depth_h}")

    smoother = DepthSmoother(max_len=DEPTH_HISTORY_LEN)
    forward_distance = 0.0     # 累计前进距离（米）
    shift_active = False       # 是否正在执行平移动作
    shift_end_time = 0.0       # 平移结束时刻

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
            print(f"[YOLO] detect num={len(detections)}, infer={(t1 - t0) * 1000:.2f} ms")

            # 收集锥桶坐标
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
                        v_cam = depth_box[3]  # 底部中心
                        x_m, y_m, z_m = pixel_to_camera_3d(u_cam, v_cam, stable_depth,
                                                           DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                        pos_3d = (x_m, y_m, z_m)
                        if cls_id == 7:
                            cones_3d.append({'x': x_m, 'y': y_m, 'z': z_m, 'depth': stable_depth})

                draw_detection(frame, color_box, cls_id, conf, stable_depth, valid_count, pos_3d)

            # --- 运动决策 ---
            # 如果正在执行一次平移，且还没结束，就继续当前平移命令，不执行新决策
            if shift_active:
                if time.time() < shift_end_time:
                    # 仍在平移中，继续发送同样的平移命令（vx=0, vy=保存的vy）
                    # 这里我们维持 last_time 很短，所以每帧都会重新发，但速度不变
                    mover.move(0, saved_vy, 0, last_time=CONTROL_PERIOD)
                    # 平移期间不累计前进距离
                    print("[Move] shifting...")
                else:
                    # 平移结束，清除标志
                    shift_active = False
                    # 恢复直行（下一帧会重新决策）
            else:
                # 正常决策
                if cones_3d:
                    nearest = min(cones_3d, key=lambda c: c['z'])
                    x, y, z = nearest['x'], nearest['y'], nearest['z']
                    vx, vy, vz, action = compute_avoidance_action(x, z)
                    print(f"[Avoid] cone x={x:.3f} z={z:.3f} -> {action} "
                          f"vx={vx} vy={vy} vz={vz}")
                else:
                    vx, vy, vz, action = VX_NOMINAL, 0, 0, "forward_no_cone"
                    print("[Avoid] no cone, straight")

                # 如果动作要求平移（vy != 0），则启动一次固定时长的平移
                if abs(vy) > 0:
                    shift_active = True
                    shift_end_time = time.time() + SHIFT_DURATION
                    saved_vy = vy
                    # 立即发送一次移动命令并进入平移状态
                    mover.move(0, vy, 0, last_time=CONTROL_PERIOD)
                    print(f"[Move] start shift: vy={vy} for {SHIFT_DURATION}s")
                else:
                    # 直行，发送指令并累计距离
                    mover.move(vx, 0, 0, last_time=CONTROL_PERIOD)
                    # 根据 vx 的正负和速度标定累计前进距离
                    # 只有向前（vx > 0）才计入正距离
                    if vx > 0:
                        forward_distance += REAL_SPEED_VX * CONTROL_PERIOD
                    elif vx < 0:
                        forward_distance -= REAL_SPEED_VX * CONTROL_PERIOD  # 后退扣减
                    print(f"[Move] straight vx={vx}, total forward: {forward_distance:.3f}m")

            # 显示与退出
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print("[User] quit by key")
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("[User] interrupted")
    finally:
        print(f"[Exit] total forward distance: {forward_distance:.3f}m")
        print("[Exit] stopping robot and camera...")
        mover.stop()
        time.sleep(0.5)
        cam.stop()
        cv2.destroyAllWindows()
        print("[Exit] done")

if __name__ == "__main__":
    main()