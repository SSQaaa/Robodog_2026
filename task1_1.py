#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4m × 1.5m 锥桶避障穿越 —— DWA 最终版
- 使用 Orbbec 深度相机 + YOLOv5-TRT 检测锥桶（cls=7）
- 采用 DWA 实时规划，自动避开锥桶与通道边界
- 保留极近距离应急后退
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
# 可调参数区
# =========================

# --- 检测相关 ---
CONF_THRESH = 0.4
MIN_VALID_DEPTH_COUNT = 20
DEPTH_HISTORY_LEN = 5

# --- 任务参数 ---
TARGET_DISTANCE = 4.0           # 总前进距离（米）

# --- 运动标定 (务必根据实测修改) ---
VX_NOMINAL = 12000              # 正常直行速度值
REAL_SPEED_VX = 1.8             # vx=12000 对应的真实前进速度（米/秒）
SPEED_FACTOR = REAL_SPEED_VX / VX_NOMINAL   # 速度值 → 米/秒 转换系数

# 横向速度估算系数（每个 vy 单位对应的横向速度 m/s）
# 粗略值：vy=30000 时约 0.5 m/s
LATERAL_SPEED_SCALE = 0.5 / 30000.0

# 控制周期
CONTROL_PERIOD = 0.1

# 后退速度值（用于极端贴近时应急）
VX_BACK = -6000

# 显示窗口
WINDOW_NAME = "4m DWA"

# 类别名称映射
CLASS_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "Green",
    5: "MPa", 6: "Red", 7: "Traffic_cone",
    8: "dashboard", 9: "ssi"
}

# 相机内参（请勿随意修改，除非重新标定）
DEPTH_FX = 453.72
DEPTH_FY = 453.72
DEPTH_CX = 310.924
DEPTH_CY = 234.049

# =========================
# UDP 通信（保持不变）
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
        self.udp.send(0x21010D06)
        self.udp.send(0x21010300)
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
    def __init__(self, max_len=DEPTH_HISTORY_LEN):
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
        cv2.putText(frame, coord_text, (x1, min(frame.shape[0] - 5, y2 + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


# =========================
# DWA 避障规划器
# =========================
class DWA_Planner:
    def __init__(self):
        # 速度约束 (m/s)
        self.max_v = 1.8          # 最大前向速度
        self.min_v = -0.3         # 允许后退
        self.max_w = 0.5          # 最大横向速度
        self.max_acc_v = 1.5      # 前进加速度限制
        self.max_acc_w = 1.0      # 横向加速度限制
        # 采样数
        self.v_samples = 9
        self.w_samples = 7
        # 预测参数
        self.dt = 0.1             # 模拟步长 (s)
        self.predict_time = 1.2   # 预测时长 (s)
        # 安全参数
        self.robot_radius = 0.25  # 机器狗半径 (m)
        self.safe_dist = 0.3      # 额外安全裕度
        # 代价权重
        self.goal_weight = 2.0
        self.obs_weight = 5.0
        self.speed_weight = 1.0
        # 边界墙 x 坐标 (m)，通道宽 1.5m，中心在 0，狗半宽 0.2，左右各留 0.1 余量
        self.wall_left = -0.55
        self.wall_right = 0.55
        # 目标点 (x, z) 在局部坐标系下 (横向0，前方4m)
        self.goal = np.array([0.0, 4.0])
        # 上一时刻速度
        self.prev_v = 0.0
        self.prev_w = 0.0

    def plan(self, cones_3d):
        """
        输入: cones_3d 列表，每个元素 {'x':, 'z':, ...}
        返回: (vx_command, vy_command) 机器狗协议速度值
        """
        # 1. 构建障碍物点集（锥桶 + 虚拟边界墙）
        obstacles = []
        for c in cones_3d:
            if 0 < c['z'] < 5.0:
                obstacles.append([c['x'], c['z']])
        # 虚拟边界墙（离散点，间隔 0.2m）
        for z in np.arange(0.0, self.predict_time * self.max_v + 0.5, 0.2):
            obstacles.append([self.wall_left, z])
            obstacles.append([self.wall_right, z])
        obstacles = np.array(obstacles) if obstacles else np.empty((0, 2))

        # 2. 动态窗口
        v_min = max(self.min_v, self.prev_v - self.max_acc_v * self.dt)
        v_max = min(self.max_v, self.prev_v + self.max_acc_v * self.dt)
        w_min = max(-self.max_w, self.prev_w - self.max_acc_w * self.dt)
        w_max = min(self.max_w, self.prev_w + self.max_acc_w * self.dt)

        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0

        # 3. 遍历采样速度
        for v in np.linspace(v_min, v_max, self.v_samples):
            for w in np.linspace(w_min, w_max, self.w_samples):
                # 轨迹模拟
                x, z = 0.0, 0.0
                min_dist = float('inf')
                steps = int(self.predict_time / self.dt)
                for _ in range(steps):
                    x += w * self.dt
                    z += v * self.dt
                    if len(obstacles) > 0:
                        dists = np.hypot(obstacles[:, 0] - x, obstacles[:, 1] - z)
                        min_dist = min(min_dist, np.min(dists))

                # 碰撞检查
                if min_dist < self.robot_radius + self.safe_dist:
                    continue

                # 代价计算
                dist_to_goal = np.hypot(self.goal[0] - x, self.goal[1] - z)
                goal_score = 1.0 / (dist_to_goal + 0.01)
                obs_score = 1.0 / (min_dist - self.robot_radius + 0.01)
                speed_score = v / self.max_v

                total = (self.goal_weight * goal_score +
                         self.obs_weight * obs_score +
                         self.speed_weight * speed_score)

                if total > best_score:
                    best_score = total
                    best_v, best_w = v, w

        # 4. 更新上一时刻速度
        self.prev_v = best_v
        self.prev_w = best_w

        # 5. 转换为机器狗指令值
        vx_cmd = int(best_v / SPEED_FACTOR)
        vy_cmd = int(best_w / LATERAL_SPEED_SCALE)

        vx_cmd = np.clip(vx_cmd, -40000, 40000)
        vy_cmd = np.clip(vy_cmd, -40000, 40000)

        return vx_cmd, vy_cmd


# =========================
# 主程序
# =========================
def main():
    print("[TRT] loading engine...")
    detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
    print("[TRT] engine loaded")

    mover = RobotMover()
    time.sleep(1)
    mover.stop()
    print("[Robot] ready (请确认狗已用手柄站立，且位于宽度中间)")

    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)
    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()
    print(f"[Orbbec] color {color_w}x{color_h} depth {depth_w}x{depth_h}")

    smoother = DepthSmoother(max_len=DEPTH_HISTORY_LEN)
    planner = DWA_Planner()

    forward_distance = 0.0
    frame_id = 0

    try:
        while forward_distance < TARGET_DISTANCE:
            # ---------- 感知 ----------
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
                        v_cam = depth_box[3]  # 用框底部作为深度采样点（更贴近地面）
                        x_m, y_m, z_m = pixel_to_camera_3d(u_cam, v_cam, stable_depth,
                                                           DEPTH_FX, DEPTH_FY,
                                                           DEPTH_CX, DEPTH_CY)
                        pos_3d = (x_m, y_m, z_m)
                        if cls_id == 7:   # 锥形桶
                            cones_3d.append({'x': x_m, 'y': y_m, 'z': z_m,
                                             'depth': stable_depth})
                draw_detection(frame, color_box, cls_id, conf,
                               stable_depth, valid_count, pos_3d)

            # ---------- 应急后退（极近距离卡死） ----------
            emergency_flag = False
            if cones_3d:
                for c in cones_3d:
                    if c['z'] < 0.5 and abs(c['x']) < 0.35:
                        emergency_flag = True
                        nearest = min(cones_3d, key=lambda c: c['z'])
                        vy_retreat = 25000 if nearest['x'] < 0 else -25000
                        print(f"[Emergency] Too close! Retreating + shift")
                        mover.move(VX_BACK, vy_retreat, 0, last_time=0.6)
                        forward_distance += VX_BACK * SPEED_FACTOR * 0.6
                        break
            if emergency_flag:
                continue   # 跳过本次 DWA，重新感知

            # ---------- DWA 决策 ----------
            target_vx, target_vy = planner.plan(cones_3d)

            # ---------- 执行 ----------
            mover.move(target_vx, target_vy, 0, last_time=CONTROL_PERIOD)

            # ---------- 里程更新（开环，仅用于判断终点） ----------
            forward_distance += target_vx * SPEED_FACTOR * CONTROL_PERIOD

            print(f"[Move] vx={target_vx} vy={target_vy} forward={forward_distance:.3f}m")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print("[User] quit by key")
                break
            frame_id += 1

    except KeyboardInterrupt:
        print("[User] interrupted")
    finally:
        print(f"[Exit] forward_distance={forward_distance:.3f}m")
        mover.stop()
        time.sleep(0.5)
        cam.stop()
        cv2.destroyAllWindows()
        print("[Exit] done")


if __name__ == "__main__":
    main()