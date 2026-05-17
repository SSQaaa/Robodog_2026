#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4m × 1.5m 锥桶避障穿越 —— DWA + 持续运动控制 + 停滞检测
"""

import os, sys, time, ctypes, threading, socket, struct
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
CONF_THRESH = 0.4
MIN_VALID_DEPTH_COUNT = 20
DEPTH_HISTORY_LEN = 5

TARGET_DISTANCE = 4.0
MAX_TIME = 60.0

VX_NOMINAL = 12000
REAL_SPEED_VX = 1.8
SPEED_FACTOR = REAL_SPEED_VX / VX_NOMINAL

LATERAL_SPEED_SCALE = 0.5 / 30000.0

MIN_EFFECTIVE_VX = 8000    # 提高最小有效前进速度
CONTROL_PERIOD = 0.05       # 指令发送周期（线程内部使用）

VX_BACK = -6000

WINDOW_NAME = "4m DWA Continuous"

CLASS_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "Green",
    5: "MPa", 6: "Red", 7: "Traffic_cone",
    8: "dashboard", 9: "ssi"
}

DEPTH_FX = 478.547
DEPTH_FY = 478.547
DEPTH_CX = 321.087
DEPTH_CY = 201.625

CONE_REAL_WIDTH_MM = 320.0
VISUAL_DEPTH_TOLERANCE = 0.4

# =========================
# UDP 通信
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

    def send(self, code, value=0, type=0):
        data = struct.pack("<3i", code, value, type)
        self.sock.sendto(data, self.send_addr)


class RobotMover:
    def __init__(self, ip='192.168.1.120', port=43893):
        self.udp = UDPClient(ip, port)
        self.target_vx = 0
        self.target_vy = 0
        self.target_vz = 0
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        # 启动心跳
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat(self):
        while True:
            self.udp.send(0x21040001)
            time.sleep(0.2)

    def set_velocity(self, vx, vy, vz=0):
        with self.lock:
            self.target_vx = int(vx)
            self.target_vy = int(vy)
            self.target_vz = int(vz)

    def _command_loop(self):
        # 确保进入移动模式
        self.udp.send(0x21010D06)
        self.udp.send(0x21010300)
        while self._running:
            with self.lock:
                vx = self.target_vx
                vy = self.target_vy
                vz = self.target_vz
            # 持续发送当前目标速度
            self.udp.send(0x21010130, vx)
            self.udp.send(0x21010131, vy)
            self.udp.send(0x21010135, vz)
            time.sleep(CONTROL_PERIOD)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._command_loop, daemon=True)
            self._thread.start()

    def stop_robot(self):
        self.set_velocity(0, 0, 0)
        time.sleep(0.2)
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.udp.send(0x21010C0A, value=7)

    def __del__(self):
        self.stop_robot()


# =========================
# 工具函数（保持不变）
# =========================
class DepthSmoother:
    # 同前...
    def __init__(self, max_len=DEPTH_HISTORY_LEN):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, obj_id, depth_mm):
        if depth_mm <= 0:
            return None
        self.history[obj_id].append(depth_mm)
        return int(np.median(np.array(self.history[obj_id])))


def scale_box(box, src_size, dst_size):
    # 同前...
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
    # 同前...
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


def draw_detection(frame, box, cls_id, conf, depth_mm, valid_count, pos_3d=None, use_visual=False):
    # 同前...
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
    if use_visual:
        cv2.putText(frame, "VIS", (x1, max(20, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# =========================
# DWA 规划器（适当调整参数）
# =========================
class DWA_Planner:
    def __init__(self):
        self.max_v = 1.8
        self.min_v = -0.3
        self.max_w = 0.5
        self.max_total_speed = 1.8
        self.max_acc_v = 1.5
        self.max_acc_w = 1.0
        self.v_samples = 9
        self.w_samples = 7
        self.dt = 0.1
        self.predict_time = 1.5        # 加长预测，提前规划绕行
        self.robot_radius = 0.25
        self.safe_dist = 0.1
        self.wall_left = -0.75
        self.wall_right = 0.75
        self.goal_weight = 3.0
        self.obs_weight = 5.0
        self.speed_weight = 1.0
        self.goal = np.array([0.0, 4.0])
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.min_physical_v = MIN_EFFECTIVE_VX * SPEED_FACTOR

    def plan(self, cones_3d, forward_distance):
        remain = max(0.0, TARGET_DISTANCE - forward_distance)
        self.goal = np.array([0.0, remain])

        obstacles = []
        for c in cones_3d:
            if 0 < c['z'] < 5.0:
                obstacles.append([c['x'], c['z']])
        wall_z_max = self.goal[1] + 2.0
        for z in np.arange(0.0, wall_z_max, 0.2):
            obstacles.append([self.wall_left, z])
            obstacles.append([self.wall_right, z])
        obstacles = np.array(obstacles) if obstacles else np.empty((0, 2))

        v_min = max(self.min_v, self.prev_v - self.max_acc_v * self.dt)
        v_max = min(self.max_v, self.prev_v + self.max_acc_v * self.dt)
        w_min = max(-self.max_w, self.prev_w - self.max_acc_w * self.dt)
        w_max = min(self.max_w, self.prev_w + self.max_acc_w * self.dt)

        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0

        for v in np.linspace(v_min, v_max, self.v_samples):
            for w in np.linspace(w_min, w_max, self.w_samples):
                if np.hypot(v, w) > self.max_total_speed:
                    continue
                x, z = 0.0, 0.0
                min_dist = float('inf')
                steps = int(self.predict_time / self.dt)
                for _ in range(steps):
                    x += w * self.dt
                    z += v * self.dt
                    if len(obstacles) > 0:
                        dists = np.hypot(obstacles[:, 0] - x, obstacles[:, 1] - z)
                        min_dist = min(min_dist, np.min(dists))
                if min_dist < self.robot_radius + self.safe_dist:
                    continue
                dist_to_goal = np.hypot(self.goal[0] - x, self.goal[1] - z)
                goal_score = 1.0 / (dist_to_goal + 0.5)
                obs_cost = -1.0 / (min_dist - self.robot_radius + 0.01)
                speed_score = v / self.max_v
                total = (self.goal_weight * goal_score +
                         self.obs_weight * obs_cost +
                         self.speed_weight * speed_score)
                if total > best_score:
                    best_score = total
                    best_v, best_w = v, w

        if best_score == -float('inf'):
            print("[DWA] No feasible path, fallback: retreat + bias")
            best_v = -0.2
            # 向更空旷的一侧平移
            if cones_3d:
                left_min = min([c['z'] for c in cones_3d if c['x'] < 0], default=5.0)
                right_min = min([c['z'] for c in cones_3d if c['x'] > 0], default=5.0)
                best_w = 0.3 if left_min < right_min else -0.3
            else:
                best_w = 0.0
        else:
            if best_v > 0 and best_v < self.min_physical_v:
                best_v = self.min_physical_v

        self.prev_v = best_v
        self.prev_w = best_w

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
    mover.stop_robot()
    print("[Robot] ready (请确认狗已用手柄站立，且位于宽度中间)")
    input("按 Enter 开始任务...")   # 给你手动准备的时间

    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)
    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()
    print(f"[Orbbec] color {color_w}x{color_h} depth {depth_w}x{depth_h}")

    smoother = DepthSmoother()
    planner = DWA_Planner()

    mover.start()   # 启动持续指令线程
    forward_distance = 0.0
    frame_id = 0
    start_time = time.time()

    # 停滞检测变量
    last_cone_z = None
    stall_counter = 0
    STALL_THRESH = 10    # 连续 10 帧锥桶距离未减小则判定为停滞

    try:
        while forward_distance < TARGET_DISTANCE:
            if time.time() - start_time > MAX_TIME:
                print("[Timeout] Task aborted")
                break

            frame = cam.get_color_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = np.asarray(frame, dtype=np.uint8).copy()

            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            print(f"[YOLO] detect num={len(detections)}, infer={(t1 - t0) * 1000:.2f} ms")

            cones_3d = []
            for det in detections:
                cx_y, cy_y, w_y, h_y, conf, cls_id = det
                if conf < CONF_THRESH:
                    continue
                color_box = yolo_to_original((cx_y, cy_y, w_y, h_y), img_w=color_w, img_h=color_h)
                depth_box = scale_box(color_box, (color_w, color_h), (depth_w, depth_h))
                dx1, dy1, dx2, dy2 = depth_box
                box_w = dx2 - dx1

                raw_depth, valid_count = cam.get_depth_in_box(*depth_box)
                stable_depth = None
                if raw_depth > 0 and valid_count >= MIN_VALID_DEPTH_COUNT:
                    stable_depth = smoother.update(int(cls_id), raw_depth)

                d_vis = None
                if cls_id == 7 and box_w > 0:
                    d_vis = (CONE_REAL_WIDTH_MM * DEPTH_FX) / box_w

                final_depth = None
                use_visual = False
                if cls_id == 7:
                    if stable_depth is not None and d_vis is not None:
                        relative_diff = abs(stable_depth - d_vis) / max(d_vis, 1.0)
                        if relative_diff > VISUAL_DEPTH_TOLERANCE:
                            final_depth = d_vis
                            use_visual = True
                        else:
                            final_depth = stable_depth
                    elif d_vis is not None:
                        final_depth = d_vis
                        use_visual = True
                    elif stable_depth is not None:
                        final_depth = stable_depth
                else:
                    final_depth = stable_depth

                if final_depth is not None:
                    u_cam = (dx1 + dx2) // 2
                    v_cam = int(dy1 + (dy2 - dy1) * 0.8)
                    x_m, y_m, z_m = pixel_to_camera_3d(u_cam, v_cam, final_depth,
                                                       DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                    pos_3d = (x_m, y_m, z_m)
                    if cls_id == 7:
                        cones_3d.append({'x': x_m, 'y': y_m, 'z': z_m, 'depth': final_depth, 'visual': use_visual})
                else:
                    pos_3d = None

                draw_detection(frame, color_box, cls_id, conf, final_depth, valid_count, pos_3d, use_visual)

            if cones_3d:
                for i, c in enumerate(cones_3d):
                    print(f"  Cone[{i}]: x={c['x']:.3f}, z={c['z']:.3f}, visual={c['visual']}")

            # ---------- 停滞检测 ----------
            if cones_3d:
                # 取最近锥桶的 z 作为移动参考
                nearest_z = min(c['z'] for c in cones_3d)
                if last_cone_z is not None:
                    if nearest_z >= last_cone_z - 0.05:   # 没有明显靠近
                        stall_counter += 1
                        if stall_counter >= STALL_THRESH:
                            print("[WARNING] Robot appears stalled! Increasing speed...")
                            # 临时提高目标速度，打破停滞
                            stall_counter = 0
                            mover.set_velocity(MIN_EFFECTIVE_VX * 2, 0)  # 加大油门
                            time.sleep(0.5)  # 给机器人一点时间响应
                    else:
                        stall_counter = 0
                last_cone_z = nearest_z
            else:
                stall_counter = 0   # 没有锥桶，无法判断

            # ---------- DWA 决策 ----------
            target_vx, target_vy = planner.plan(cones_3d, forward_distance)

            # 钳位速度到有效范围
            if 0 < target_vx < MIN_EFFECTIVE_VX:
                target_vx = MIN_EFFECTIVE_VX
            elif -MIN_EFFECTIVE_VX < target_vx < 0:
                target_vx = -MIN_EFFECTIVE_VX

            # 设置持续目标速度
            mover.set_velocity(target_vx, target_vy)

            # 里程累积（注意：这只是估计，真实位置我们依靠视觉停滞检测修正）
            forward_distance += target_vx * SPEED_FACTOR * 0.1  # 假设0.1秒规划周期

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
        mover.stop_robot()
        time.sleep(0.5)
        cam.stop()
        cv2.destroyAllWindows()
        print("[Exit] done")


if __name__ == "__main__":
    main()
