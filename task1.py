#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4m × 1.7m 锥桶避障穿越（最终修复版）
- 修正后退里程计算 bug
- 避障平移持续到 z > 安全距离
- 统一速度换算
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
# 可调参数区（请根据实际测试情况修改）
# =========================

# --- 检测相关 ---
CONF_THRESH = 0.4               # YOLO 检测置信度阈值，低于此值的检测框将被忽略
MIN_VALID_DEPTH_COUNT = 20      # 框内有效深度点的最少数量，低于此值认为深度无效
DEPTH_HISTORY_LEN = 5           # 深度平滑队列长度，用于取中值滤波，减少抖动

# --- 任务与避障逻辑 ---
TARGET_DISTANCE = 4.0           # 总前进距离（米），到达后自动停止
SAFE_DISTANCE = 1.40             # 开始触发避障的前向距离（米），锥桶 z < 此值才考虑避障
LATERAL_THRESHOLD = 0.38        # 横向判定阈值（米），abs(x) < 此值认为锥桶在正前方
CRITICAL_DISTANCE = 0.7         # 紧急后退触发距离（米），z < 此值且正前方则强制后退+平移

# --- 运动指令速度值 (原始单位，与机器狗协议匹配) ---
VX_NOMINAL = 12000              # 正常直行速度值
VY_SHIFT = 25000                # 避障平移速度值（正值向右，负值向左）
VX_BACK = -6000                 # 后退速度值（负值表示后退）
VX_SLOW = 8000                  # 减速直行速度值（用于无法避障时缓慢通过）

# --- 速度标定 (实际物理速度) ---
REAL_SPEED_VX = 1.8             # vx=12000 对应的真实前进速度（米/秒），需实测校准
SPEED_FACTOR = REAL_SPEED_VX / VX_NOMINAL   # 速度值 → 米/秒 的转换系数（自动计算）

# 横向速度估算系数（每个 vy 单位对应的横向速度 m/s）
# 粗略值：vy=30000 时约 0.5 m/s，可按比例调整
LATERAL_SPEED_SCALE = 0.5 / 30000.0

# --- 边界保护参数 ---
MAX_LATERAL_OFFSET = 0.45        # 允许的最大横向偏移（米），超过此值会触发边界修正
BOUNDARY_CORRECTION_VY = 25000  # 边界修正时的横向平移速度（正值向右）
BOUNDARY_EMERGENCY = 0.50       # 紧急边界偏移（米），超过此值立即后退+向中心平移

# --- 控制周期 ---
CONTROL_PERIOD = 0.1            # 运动指令的持续时间（秒），一般无需修改

# 显示窗口名称
WINDOW_NAME = "4m Final Fix"

# 类别名称映射（与你的模型一致）
CLASS_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "Green",
    5: "MPa", 6: "Red", 7: "Traffic_cone",
    8: "dashboard", 9: "ssi"
}

# 相机内参（来自你的日志，请勿随意修改，除非相机重新标定）
DEPTH_FX = 453.72
DEPTH_FY = 453.72
DEPTH_CX = 310.924
DEPTH_CY = 234.049

# =========================
# UDP 通信（不变）
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
        if depth_mm <= 0: return None
        self.history[obj_id].append(depth_mm)
        return int(np.median(np.array(self.history[obj_id])))

def scale_box(box, src_size, dst_size):
    x1,y1,x2,y2 = box
    src_w,src_h = src_size
    dst_w,dst_h = dst_size
    dx1 = int(round(x1*dst_w/float(src_w)))
    dy1 = int(round(y1*dst_h/float(src_h)))
    dx2 = int(round(x2*dst_w/float(src_w)))
    dy2 = int(round(y2*dst_h/float(src_h)))
    dx1 = max(0, min(dst_w-1, dx1))
    dx2 = max(0, min(dst_w-1, dx2))
    dy1 = max(0, min(dst_h-1, dy1))
    dy2 = max(0, min(dst_h-1, dy2))
    return dx1,dy1,dx2,dy2

def yolo_to_original(box, img_w, img_h, input_size=640):
    cx,cy,w,h = box
    scale = min(input_size/img_w, input_size/img_h)
    new_w = img_w*scale
    new_h = img_h*scale
    pad_x = (input_size-new_w)/2
    pad_y = (input_size-new_h)/2
    cx = (cx-pad_x)/scale
    cy = (cy-pad_y)/scale
    w = w/scale; h = h/scale
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = int(cx + w/2); y2 = int(cy + h/2)
    x1 = max(0, min(img_w-1,x1)); x2 = max(0, min(img_w-1,x2))
    y1 = max(0, min(img_h-1,y1)); y2 = max(0, min(img_h-1,y2))
    return x1,y1,x2,y2

def pixel_to_camera_3d(u,v,depth_mm,fx,fy,cx,cy):
    z = depth_mm/1000.0
    x = (u-cx)*z/fx
    y = (v-cy)*z/fy
    return x,y,z

def draw_detection(frame, box, cls_id, conf, depth_mm, valid_count, pos_3d=None):
    x1,y1,x2,y2 = box
    cls_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")
    if depth_mm is not None and depth_mm>0:
        text = f"{cls_name} {conf:.2f} {depth_mm/1000.0:.2f}m"
    else:
        text = f"{cls_name} {conf:.2f} Depth Invalid"
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(frame, text, (x1,max(25,y1-10)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame, f"ValidPts: {valid_count}", (x1, min(frame.shape[0]-5, y2+20)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    if pos_3d is not None:
        x_3d,y_3d,z_3d = pos_3d
        coord_text = f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}m"
        cv2.putText(frame, coord_text, (x1, min(frame.shape[0]-5, y2+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

# =========================
# 避障辅助
# =========================
def get_avoid_direction(x, z):
    if z > SAFE_DISTANCE or abs(x) >= LATERAL_THRESHOLD:
        return 0
    return 1 if x < 0 else -1

def will_exceed_limit(direction, current_lateral):
    delta = direction * VY_SHIFT * LATERAL_SPEED_SCALE * CONTROL_PERIOD
    return abs(current_lateral + delta) > MAX_LATERAL_OFFSET

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
    depth_w,depth_h = cam.get_depth_size()
    color_w,color_h = cam.get_color_size()
    print(f"[Orbbec] color {color_w}x{color_h} depth {depth_w}x{depth_h}")

    smoother = DepthSmoother(max_len=DEPTH_HISTORY_LEN)
    forward_distance = 0.0
    lateral_offset = 0.0
    frame_id = 0

    avoid_active = False
    avoid_direction = 0

    try:
        while forward_distance < TARGET_DISTANCE:
            frame = cam.get_color_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = np.asarray(frame, dtype=np.uint8).copy()
            color_h,color_w = frame.shape[:2]

            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            print(f"[YOLO] detect num={len(detections)}, infer={(t1-t0)*1000:.2f} ms")

            cones_3d = []
            for det in detections:
                cx_y,cy_y,w_y,h_y,conf,cls_id = det
                if conf < CONF_THRESH: continue
                color_box = yolo_to_original((cx_y,cy_y,w_y,h_y), img_w=color_w,img_h=color_h)
                depth_box = scale_box(color_box, (color_w,color_h), (depth_w,depth_h))
                raw_depth,valid_count = cam.get_depth_in_box(*depth_box)
                stable_depth = None
                pos_3d = None
                if raw_depth>0 and valid_count>=MIN_VALID_DEPTH_COUNT:
                    stable_depth = smoother.update(int(cls_id), raw_depth)
                    if stable_depth is not None:
                        u_cam = (depth_box[0]+depth_box[2])//2
                        v_cam = depth_box[3]
                        x_m,y_m,z_m = pixel_to_camera_3d(u_cam,v_cam,stable_depth,
                                                         DEPTH_FX,DEPTH_FY,DEPTH_CX,DEPTH_CY)
                        pos_3d = (x_m,y_m,z_m)
                        if cls_id == 7:
                            cones_3d.append({'x':x_m,'y':y_m,'z':z_m,'depth':stable_depth})
                draw_detection(frame, color_box, cls_id, conf, stable_depth, valid_count, pos_3d)

            # 决策
            target_vx = VX_NOMINAL
            target_vy = 0

            # 1. 紧急边界保护
            if abs(lateral_offset) > BOUNDARY_EMERGENCY:
                target_vx = VX_BACK
                target_vy = -BOUNDARY_CORRECTION_VY if lateral_offset > 0 else BOUNDARY_CORRECTION_VY
                avoid_active = False
                print(f"[Boundary Emergency] lateral={lateral_offset:.3f}, retreating")
            # 2. 常规边界修正（不在避障中）
            elif abs(lateral_offset) > MAX_LATERAL_OFFSET and not avoid_active:
                target_vx = 0
                target_vy = -BOUNDARY_CORRECTION_VY if lateral_offset > 0 else BOUNDARY_CORRECTION_VY
                print(f"[Boundary Correct] lateral={lateral_offset:.3f}")
            # 3. 锥桶避障
            elif cones_3d:
                nearest = min(cones_3d, key=lambda c: c['z'])
                x, z = nearest['x'], nearest['z']

                if z <= CRITICAL_DISTANCE and abs(x) < LATERAL_THRESHOLD:
                    target_vx = VX_BACK
                    target_vy = VY_SHIFT if x < 0 else -VY_SHIFT
                    if will_exceed_limit(target_vy, lateral_offset):
                        target_vy = -target_vy
                    avoid_active = False
                    print(f"[Emergency] back+shift, vy={target_vy}")
                else:
                    direction = get_avoid_direction(x, z)
                    if avoid_active:
                        if direction != 0 and direction == avoid_direction and z <= SAFE_DISTANCE:
                            if will_exceed_limit(avoid_direction, lateral_offset):
                                new_dir = -avoid_direction
                                if not will_exceed_limit(new_dir, lateral_offset):
                                    avoid_direction = new_dir
                                else:
                                    avoid_active = False
                                    target_vx = VX_SLOW
                                    target_vy = 0
                                    print("[Avoid] boundary conflict, slow forward")
                            if avoid_active:
                                target_vx = 0
                                target_vy = VY_SHIFT if avoid_direction == 1 else -VY_SHIFT
                                print(f"[Avoid] continue shift {avoid_direction}, cone x={x:.3f} z={z:.3f}")
                        else:
                            avoid_active = False
                            print(f"[Avoid] stopped (z={z:.2f})")
                    else:
                        if direction != 0:
                            if will_exceed_limit(direction, lateral_offset):
                                opp_dir = -direction
                                if not will_exceed_limit(opp_dir, lateral_offset):
                                    direction = opp_dir
                                else:
                                    direction = 0
                                    target_vx = VX_SLOW
                                    target_vy = 0
                                    print("[Avoid] cannot shift, slow forward")
                            if direction != 0:
                                avoid_active = True
                                avoid_direction = direction
                                target_vx = 0
                                target_vy = VY_SHIFT if direction == 1 else -VY_SHIFT
                                print(f"[Avoid] start shift {'right' if direction==1 else 'left'}, cone x={x:.3f}")
            else:
                avoid_active = False

            # 执行统一指令
            mover.move(target_vx, target_vy, 0, last_time=CONTROL_PERIOD)

            # 里程更新（修复 bug）
            forward_distance += target_vx * SPEED_FACTOR * CONTROL_PERIOD
            lateral_offset += target_vy * LATERAL_SPEED_SCALE * CONTROL_PERIOD

            state = "shift" if avoid_active else "straight"
            print(f"[Move] {state} vx={target_vx} vy={target_vy} "
                  f"forward={forward_distance:.3f}m lateral={lateral_offset:.3f}m")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key==27 or key==ord('q'):
                print("[User] quit by key")
                break
            frame_id += 1

    except KeyboardInterrupt:
        print("[User] interrupted")
    finally:
        print(f"[Exit] forward_distance={forward_distance:.3f}m, lateral_offset={lateral_offset:.3f}m")
        mover.stop()
        time.sleep(0.5)
        cam.stop()
        cv2.destroyAllWindows()
        print("[Exit] done")

if __name__ == "__main__":
    main()
