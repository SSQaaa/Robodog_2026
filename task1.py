#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4m锥桶避障穿越 - 逐个处理版（基于车身真实宽度 + 有序穿越）
"""

import os, sys, time, ctypes, threading, socket, struct
from collections import defaultdict, deque
import cv2
import numpy as np
import orbbec_native

# ========================= 路径配置 =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LIB_DIR = os.path.join(BASE_DIR, "/home/ysc/Desktop/2026Project/libs/")
ENGINE_PATH = os.path.join(TRT_LIB_DIR, "bigdog_0427.engine")
sys.path.append(TRT_LIB_DIR)
ctypes.CDLL(os.path.join(TRT_LIB_DIR, "libmyplugins.so"))
import yolov5_trt_cpp

# ========================= 可调参数 =========================
CONF_THRESH = 0.4
MIN_VALID_DEPTH_COUNT = 10
DEPTH_HISTORY_LEN = 5
VISUAL_DEPTH_HISTORY_LEN = 5

TARGET_DISTANCE = 3.7
SAFE_DISTANCE = 0.83          # 开始考虑避障的纵向距离阈值（大于此距离的阻塞锥桶可先忽略）
CRITICAL_DISTANCE = 0.7       # 极近距离，必须立即处理

# 降级模式参数
DEPTH_FAILURE_THRESH = 10
SAFE_DISTANCE_FALLBACK = 1.0
CRITICAL_DISTANCE_FALLBACK = 0.5

# 速度参数
VX_NOMINAL = 8000
VX_AVOID = 3000               # 避障平移时的前向速度（可微调）
VX_BACKWARD = -8000
VY_SHIFT = 25000
VY_SHIFT_EMERGENCY = 35000

SPEED_FACTOR = 0.000075       # 每单位 vx 对应的实际速度 (m/s) / (控制值)
LATERAL_SPEED_SCALE = 2.574e-5   # 横向速度系数（已校准）
MAX_LATERAL_OFFSET = 0.40
BOUNDARY_CORRECTION_VY = 25000
BOUNDARY_EMERGENCY = 0.50

CONTROL_PERIOD = 0.1

ROBOT_LENGTH = 0.7
ROBOT_HALF_WIDTH = 0.37       # 机器人半宽（根据实际车宽调整）
LATERAL_SAFETY_MARGIN = 0.05  # 横向安全余量
PASSING_MARGIN = 0.10

NEARFIELD_THRESHOLD_MM = 350
LATERAL_DEAD_ZONE = 0.04

WINDOW_NAME = "FinalCalibrated Avoidance"

CLASS_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "Green",
    5: "MPa", 6: "Red", 7: "Traffic_cone",
    8: "dashboard", 9: "ssi"
}

# 相机内参
DEPTH_FX = 478.547
DEPTH_FY = 478.547
DEPTH_CX = 321.087
DEPTH_CY = 201.625
REAL_CONE_WIDTH = 0.32
COLOR_FX = 453.72
VISUAL_FAR_THRESH = 1.9
DEPTH_CLOSE_THRESH = 1.5

# ========================= UDP 通信 =========================
class UDPClient:
    def __init__(self, ip, port):
        self.ip = ip; self.port = port
        self.send_addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)
    def __del__(self):
        self.sock.close()
    def send(self, code, value=0, type=0, last_time=0, duration=0):
        data = struct.pack("<3i", code, value, type)
        start = time.time()
        if last_time == 0:
            self.sock.sendto(data, self.send_addr)
            time.sleep(0.05)
        else:
            while time.time() - start < last_time:
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
    def stand_up(self, duration=3):
        self.udp.send(0x21010202, duration=duration)
    def revolve_90(self):
        self.udp.send(0x21010C0A, value=14)
        time.sleep(2)
        self.udp.send(0x21010C0A, value=7)
        time.sleep(0.5)
    def stop(self):
        self.udp.send(0x21010C0A, value=7)
    def __del__(self):
        self.stop()

# ========================= 简易 IoU 跟踪器 =========================
class SimpleConeTracker:
    def __init__(self, max_age=5, iou_thresh=0.3):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.iou_thresh = iou_thresh
    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    def update(self, detections):
        matched_track_ids = set()
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return {}
        if not self.tracks:
            result = {}
            for det in detections:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'box': det['color_box'], 'age':0, 'visual_z':det['visual_z']}
                matched_track_ids.add(tid)
                result[tid] = det
            return result
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                cost_matrix[i][j] = 1.0 - self._iou(self.tracks[tid]['box'], det['color_box'])
        det_used = set()
        pairs = []
        while True:
            min_cost = 1.0; best_t = -1; best_d = -1
            for i, tid in enumerate(track_ids):
                if tid in matched_track_ids: continue
                for j in range(len(detections)):
                    if j in det_used: continue
                    if cost_matrix[i][j] < min_cost:
                        min_cost = cost_matrix[i][j]; best_t = i; best_d = j
            if best_t == -1 or min_cost > (1 - self.iou_thresh):
                break
            tid = track_ids[best_t]
            self.tracks[tid]['box'] = detections[best_d]['color_box']
            self.tracks[tid]['visual_z'] = detections[best_d]['visual_z']
            self.tracks[tid]['age'] = 0
            matched_track_ids.add(tid); det_used.add(best_d)
            pairs.append((tid, detections[best_d]))
        for j, det in enumerate(detections):
            if j not in det_used:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'box': det['color_box'], 'age':0, 'visual_z':det['visual_z']}
                matched_track_ids.add(tid)
                pairs.append((tid, det))
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        return {tid: det for tid, det in pairs}

# ========================= 深度平滑器 =========================
class DepthSmoother:
    def __init__(self, max_len=DEPTH_HISTORY_LEN):
        self.history = defaultdict(lambda: deque(maxlen=max_len))
        self.latest = {}
    def update(self, obj_id, depth_mm):
        if depth_mm <= 0: return None
        self.history[obj_id].append(depth_mm)
        median = int(np.median(np.array(self.history[obj_id])))
        self.latest[obj_id] = median
        return median
    def get_latest(self, obj_id):
        return self.latest.get(obj_id, None)

class VisualDepthSmoother:
    def __init__(self, max_len=VISUAL_DEPTH_HISTORY_LEN):
        self.history = defaultdict(lambda: deque(maxlen=max_len))
    def update(self, obj_id, visual_z):
        if visual_z <= 0: return None
        self.history[obj_id].append(visual_z)
        return np.median(np.array(self.history[obj_id]))

# ========================= 工具函数 =========================
def scale_box(box, src_size, dst_size):
    x1,y1,x2,y2 = box
    src_w,src_h = src_size
    dst_w,dst_h = dst_size
    dx1 = int(round(x1*dst_w/float(src_w))); dy1 = int(round(y1*dst_h/float(src_h)))
    dx2 = int(round(x2*dst_w/float(src_w))); dy2 = int(round(y2*dst_h/float(src_h)))
    dx1 = max(0, min(dst_w-1, dx1)); dx2 = max(0, min(dst_w-1, dx2))
    dy1 = max(0, min(dst_h-1, dy1)); dy2 = max(0, min(dst_h-1, dy2))
    return dx1,dy1,dx2,dy2

def yolo_to_original(box, img_w, img_h, input_size=640):
    cx,cy,w,h = box
    scale = min(input_size/img_w, input_size/img_h)
    new_w = img_w*scale; new_h = img_h*scale
    pad_x = (input_size-new_w)/2; pad_y = (input_size-new_h)/2
    cx = (cx-pad_x)/scale; cy = (cy-pad_y)/scale
    w = w/scale; h = h/scale
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = int(cx + w/2); y2 = int(cy + h/2)
    x1 = max(0, min(img_w-1,x1)); x2 = max(0, min(img_w-1,x2))
    y1 = max(0, min(img_h-1,y1)); y2 = max(0, min(img_h-1,y2))
    return x1,y1,x2,y2

def pixel_to_camera_3d(u,v,depth_mm,fx,fy,cx,cy):
    z = depth_mm/1000.0
    x = (u-cx)*z/fx; y = (v-cy)*z/fy
    return x,y,z

def get_cone_depth_roi(color_box, color_size, depth_size):
    depth_box = scale_box(color_box, color_size, depth_size)
    dx1, dy1, dx2, dy2 = depth_box
    h = dy2 - dy1
    if h <= 0:
        return None
    target_h = max(5, h // 4)
    roi_y2 = dy2
    roi_y1 = dy2 - target_h
    if roi_y1 < 0:
        roi_y1 = 0
        roi_y2 = min(depth_size[1], roi_y1 + target_h)
    if roi_y2 - roi_y1 < 5:
        roi_y1 = dy1
        roi_y2 = dy1 + min(target_h, depth_size[1] - dy1)
    w = dx2 - dx1
    target_w = max(5, w // 2)
    center = (dx1 + dx2) // 2
    roi_x1 = max(0, center - target_w // 2)
    roi_x2 = min(depth_size[0], roi_x1 + target_w)
    if roi_x2 - roi_x1 < 5:
        roi_x1 = dx1
        roi_x2 = dx2
    return (roi_x1, roi_y1, roi_x2, roi_y2)

def draw_detection(frame, box, cls_id, conf, depth_mm, valid_count, pos_3d=None, depth_error=False, estimated=False, passed=False, depth_roi_box=None):
    x1,y1,x2,y2 = box
    cls_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")
    status = ""
    if passed: status += " [PASSED]"
    if estimated: status += " [est]"
    if depth_error:
        text = f"{cls_name} {conf:.2f} DEPTH ERROR"; color = (0,0,255)
    elif depth_mm:
        text = f"{cls_name} {conf:.2f} {depth_mm/1000.0:.2f}m{status}"
        color = (0,255,0)
    else:
        text = f"{cls_name} {conf:.2f} 深度无效"; color = (0,255,0)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if depth_roi_box is not None:
        rx1, ry1, rx2, ry2 = depth_roi_box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255,0,0), 2)
    cv2.putText(frame, text, (x1,max(25,y1-10)), cv2.FONT_HERSHEY_SIMPLEX,0.6, color, 2)
    cv2.putText(frame, f"ValidPts: {valid_count}", (x1, min(frame.shape[0]-5, y2+20)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0), 2)
    if pos_3d and not depth_error:
        x_3d,y_3d,z_3d = pos_3d
        cv2.putText(frame, f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}m",
                    (x1, min(frame.shape[0]-5, y2+40)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,0),2)

def will_exceed_limit(direction, current_lateral, vy_shift=VY_SHIFT):
    delta = direction * vy_shift * LATERAL_SPEED_SCALE * CONTROL_PERIOD
    return abs(current_lateral + delta) > MAX_LATERAL_OFFSET

def is_blocking(cone, robot_lat_offset):
    """
    判断锥桶是否在机器人的行驶路径上（即会与车身发生碰撞）
    robot_lat_offset: 机器人中心当前的横向偏移
    """
    cone_x = cone['x']  # 锥桶相对相机的x坐标（等价于全局x，假设无旋转）
    left_edge = robot_lat_offset - ROBOT_HALF_WIDTH - LATERAL_SAFETY_MARGIN
    right_edge = robot_lat_offset + ROBOT_HALF_WIDTH + LATERAL_SAFETY_MARGIN
    # 锥桶位于左右边缘之间（含余量）
    return left_edge < cone_x < right_edge

# ========================= 主程序 =========================
def main():
    print("[TRT] 正在加载引擎...")
    detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
    print("[TRT] 引擎加载完成")

    mover = RobotMover()
    time.sleep(1); mover.stop(); time.sleep(0.5)

    print("[机器人] 正在起立...")
    mover.stand_up(duration=3)
    time.sleep(0.5)
    print("[机器人] 已就绪")

    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)
    depth_w,depth_h = cam.get_depth_size()
    color_w,color_h = cam.get_color_size()
    print(f"[Orbbec] 彩色 {color_w}x{color_h} 深度 {depth_w}x{depth_h}")

    tracker = SimpleConeTracker(max_age=5, iou_thresh=0.3)
    depth_smoother = DepthSmoother(max_len=DEPTH_HISTORY_LEN)
    visual_smoother = VisualDepthSmoother(max_len=VISUAL_DEPTH_HISTORY_LEN)
    forward_distance = 0.0
    lateral_offset = 0.0
    frame_id = 0

    cone_first_seen_at = {}
    depth_failure_counter = 0
    depth_camera_healthy = True

    try:
        while forward_distance < TARGET_DISTANCE:
            frame = cam.get_color_frame()
            if frame is None:
                time.sleep(0.01); continue
            frame = np.asarray(frame, dtype=np.uint8).copy()
            color_h,color_w = frame.shape[:2]

            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            print(f"[YOLO] 检测到 {len(detections)} 个目标, 推理 {(t1-t0)*1000:.1f} ms")

            track_input = []
            for det in detections:
                cx_y,cy_y,w_y,h_y,conf,cls_id = det
                if conf < CONF_THRESH or cls_id != 7:
                    continue
                color_box = yolo_to_original((cx_y,cy_y,w_y,h_y), img_w=color_w,img_h=color_h)
                x1,y1,x2,y2 = color_box
                box_w = x2 - x1
                raw_visual_z = (REAL_CONE_WIDTH * COLOR_FX) / max(box_w, 1) if box_w > 0 else 5.0
                track_input.append({'color_box': color_box, 'visual_z': raw_visual_z})

            track_dict = tracker.update(track_input)

            cones_3d = []
            has_valid_depth_this_frame = False

            for tid, det_data in track_dict.items():
                color_box = det_data['color_box']
                raw_visual_z = det_data['visual_z']
                smoothed_visual_z = visual_smoother.update(tid, raw_visual_z)
                visual_z = smoothed_visual_z if smoothed_visual_z is not None else raw_visual_z

                if tid not in cone_first_seen_at:
                    cone_first_seen_at[tid] = forward_distance

                depth_roi = get_cone_depth_roi(color_box, (color_w,color_h), (depth_w,depth_h))
                if depth_roi is None:
                    continue

                color_roi_box = scale_box(depth_roi, (depth_w,depth_h), (color_w,color_h))
                raw_depth, valid_count = cam.get_depth_in_box(*depth_roi)

                stable_depth = None
                is_estimated = False
                z_meas = None
                pos_3d = None
                depth_error = False

                if raw_depth > 0 and valid_count >= MIN_VALID_DEPTH_COUNT:
                    stable_depth = depth_smoother.update(tid, raw_depth)
                    if stable_depth is not None:
                        u_cam = (depth_roi[0] + depth_roi[2]) // 2
                        v_cam = depth_roi[3] - 1
                        x_m, y_m, z_m = pixel_to_camera_3d(u_cam, v_cam, stable_depth,
                                                           DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                        pos_3d = (x_m, y_m, z_m)
                        z_meas = z_m
                        has_valid_depth_this_frame = True
                        if visual_z > VISUAL_FAR_THRESH and z_meas < DEPTH_CLOSE_THRESH:
                            depth_error = True
                            if visual_z > 0:
                                z_m_est = visual_z
                                u_cam = (depth_roi[0] + depth_roi[2]) // 2
                                v_cam = depth_roi[3] - 1
                                x_m_est = (u_cam - DEPTH_CX) * z_m_est / DEPTH_FX
                                y_m_est = (v_cam - DEPTH_CY) * z_m_est / DEPTH_FY
                                pos_3d = (x_m_est, y_m_est, z_m_est)
                                is_estimated = True
                                print(f"[深度错误] tid={tid} 深度={z_meas:.2f}m 视觉={visual_z:.2f}m，使用视觉估计")
                else:
                    if visual_z > 0:
                        stable_depth = int(visual_z * 1000)
                        is_estimated = True
                        u_cam = (depth_roi[0] + depth_roi[2]) // 2
                        v_cam = depth_roi[3] - 1
                        x_m = (u_cam - DEPTH_CX) * visual_z / DEPTH_FX
                        y_m = (v_cam - DEPTH_CY) * visual_z / DEPTH_FY
                        pos_3d = (x_m, y_m, visual_z)
                        z_meas = visual_z
                        print(f"[视觉估计] tid={tid}, 视觉深度={visual_z:.2f}m")
                    else:
                        continue

                # 通过判断
                passed = False
                if not is_estimated and tid in cone_first_seen_at and pos_3d is not None:
                    current_z = pos_3d[2]
                    dist_traveled = forward_distance - cone_first_seen_at[tid]
                    if dist_traveled > (current_z + ROBOT_LENGTH + PASSING_MARGIN):
                        passed = True
                        print(f"[通过] tid={tid} 已走过")
                if is_estimated:
                    passed = False

                draw_detection(frame, color_box, 7, 0.0, stable_depth, valid_count,
                               pos_3d if not depth_error else None, depth_error, is_estimated,
                               passed, color_roi_box)

                if not passed and pos_3d is not None:
                    cones_3d.append({'x':pos_3d[0], 'y':pos_3d[1], 'z':pos_3d[2],
                                     'depth':stable_depth, 'estimated':is_estimated, 'tid':tid})
                    if is_estimated:
                        print(f"[威胁列表] 添加视觉估计锥桶 tid={tid}, z={pos_3d[2]:.2f}m")

            # 更新深度相机健康状态
            if has_valid_depth_this_frame:
                depth_failure_counter = 0
                depth_camera_healthy = True
            else:
                depth_failure_counter += 1
                if depth_failure_counter >= DEPTH_FAILURE_THRESH:
                    depth_camera_healthy = False

            if not depth_camera_healthy:
                current_safe_dist = SAFE_DISTANCE_FALLBACK
                current_critical_dist = CRITICAL_DISTANCE_FALLBACK
                print(f"[降级模式] 深度相机失效 {depth_failure_counter} 帧, 保守距离: safe={current_safe_dist}m, critical={current_critical_dist}m")
            else:
                current_safe_dist = SAFE_DISTANCE
                current_critical_dist = CRITICAL_DISTANCE

            # 调试打印
            if cones_3d:
                print("[DEBUG] ===== 当前锥桶列表 =====")
                for i, c in enumerate(cones_3d):
                    print(f"  [{i}] tid={c['tid']}, x={c['x']:.3f}, z={c['z']:.3f}, est={c['estimated']}")
            else:
                print("[DEBUG] 无锥桶")

            # ========== 近场盲区保护 ==========
            target_vx = VX_NOMINAL
            target_vy = 0
            nearfield_active = False

            nf_x1 = int(depth_w * 0.25)
            nf_x2 = int(depth_w * 0.75)
            nf_y1 = max(0, depth_h - 40)
            nf_y2 = depth_h - 1
            raw_near, valid_near = cam.get_depth_in_box(nf_x1, nf_y1, nf_x2, nf_y2)
            if raw_near > 0 and valid_near > 20 and raw_near < NEARFIELD_THRESHOLD_MM:
                nearfield_active = True
                left_margin = MAX_LATERAL_OFFSET + lateral_offset
                right_margin = MAX_LATERAL_OFFSET - lateral_offset
                if left_margin > right_margin:
                    direction = -1
                elif right_margin > left_margin:
                    direction = 1
                else:
                    direction = 1
                if direction != 0 and not will_exceed_limit(direction, lateral_offset):
                    target_vx = 0
                    target_vy = VY_SHIFT if direction == 1 else -VY_SHIFT
                    print(f"[近场保护] 深度={raw_near/1000:.3f}m，向{'右' if direction==1 else '左'}平移")
                else:
                    target_vx = 0; target_vy = 0
                    print(f"[近场保护] 深度={raw_near/1000:.3f}m，无法平移，停止！")

            if nearfield_active:
                mover.move(target_vx, target_vy, 0, last_time=CONTROL_PERIOD)
                forward_distance += target_vx * SPEED_FACTOR * CONTROL_PERIOD
                lateral_offset += target_vy * LATERAL_SPEED_SCALE * CONTROL_PERIOD
                print(f"[运动] 近场保护 vx={target_vx} vy={target_vy} 前进={forward_distance:.3f}m 横向={lateral_offset:.3f}m")
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) in (27, ord('q')):
                    break
                # 清理跟踪器
                for tid in list(cone_first_seen_at.keys()):
                    if tid not in tracker.tracks:
                        del cone_first_seen_at[tid]
                frame_id += 1
                continue

            # ========== 边界修正（硬限制） ==========
            if abs(lateral_offset) > BOUNDARY_EMERGENCY:
                target_vx = 0
                target_vy = -BOUNDARY_CORRECTION_VY if lateral_offset > 0 else BOUNDARY_CORRECTION_VY
                state_str = "边界紧急修正"
                print(f"[边界紧急] 横向偏移={lateral_offset:.3f}m，停止平移修正")
            elif abs(lateral_offset) > MAX_LATERAL_OFFSET and target_vy == 0:
                target_vx = 0
                target_vy = -BOUNDARY_CORRECTION_VY if lateral_offset > 0 else BOUNDARY_CORRECTION_VY
                state_str = "边界修正"
                print(f"[边界修正] 横向偏移={lateral_offset:.3f}m，修正中")
            else:
                # ========== 核心决策：逐个处理最近阻塞锥桶 ==========
                if not cones_3d:
                    target_vx = VX_NOMINAL
                    target_vy = 0
                    state_str = "直行（无锥桶）"
                else:
                    # 按纵向距离排序
                    sorted_cones = sorted(cones_3d, key=lambda c: c['z'])
                    blocking_cone = None
                    for cone in sorted_cones:
                        if is_blocking(cone, lateral_offset):
                            blocking_cone = cone
                            break

                    if blocking_cone is None:
                        # 无阻塞锥桶，直行
                        target_vx = VX_NOMINAL
                        target_vy = 0
                        state_str = "直行（无阻塞）"
                    else:
                        z_block = blocking_cone['z']
                        x_block = blocking_cone['x']

                        if z_block > current_safe_dist:
                            # 距离尚远，暂不反应
                            target_vx = VX_NOMINAL
                            target_vy = 0
                            state_str = f"直行（阻塞锥桶距离{z_block:.2f}m > safe）"
                        else:
                            # 在安全距离内，需要处理
                            # 先判断是否可以平移避开
                            left_possible = not will_exceed_limit(-1, lateral_offset)
                            right_possible = not will_exceed_limit(1, lateral_offset)

                            # 首选远离锥桶的方向
                            if x_block > 0:
                                desired_dir = -1  # 锥桶在右侧，向左移
                            else:
                                desired_dir = 1   # 锥桶在左侧，向右移

                            if desired_dir == -1 and left_possible:
                                vy = -VY_SHIFT
                            elif desired_dir == 1 and right_possible:
                                vy = VY_SHIFT
                            else:
                                vy = 0

                            if vy != 0:
                                # 可以平移避障
                                target_vx = VX_AVOID
                                target_vy = vy
                                state_str = f"平移避障 ({'左' if vy<0 else '右'}) + 微前进"
                            else:
                                # 无法平移，判断是否可以继续直行（锥桶距离 > 车长+余量）
                                if z_block > ROBOT_LENGTH + 0.2:
                                    target_vx = VX_NOMINAL
                                    target_vy = 0
                                    state_str = f"直行通过（锥桶{z_block:.2f}m，>车身长）"
                                else:
                                    # 极近且无法平移 → 紧急后退或停止
                                    if z_block <= current_critical_dist:
                                        target_vx = VX_BACKWARD
                                        target_vy = 0
                                        state_str = f"紧急后退（锥桶{z_block:.2f}m < critical）"
                                    else:
                                        target_vx = 0
                                        target_vy = 0
                                        state_str = f"停止（无法平移，距离{z_block:.2f}m）"

            # 清理过期的跟踪记录
            for tid in list(cone_first_seen_at.keys()):
                if tid not in tracker.tracks:
                    del cone_first_seen_at[tid]

            # 执行运动
            mover.move(target_vx, target_vy, 0, last_time=CONTROL_PERIOD)
            forward_distance += target_vx * SPEED_FACTOR * CONTROL_PERIOD
            lateral_offset += target_vy * LATERAL_SPEED_SCALE * CONTROL_PERIOD

            print(f"[运动] {state_str} vx={target_vx} vy={target_vy} 前进={forward_distance:.3f}m 横向={lateral_offset:.3f}m")

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) in (27, ord('q')):
                print("[用户] 退出键"); break
            frame_id += 1

        # 横向归正
        print(f"[任务] 4米已走完，当前横向偏移={lateral_offset:.3f}m，开始回正...")
        TOLERANCE = 0.02
        if abs(lateral_offset) > TOLERANCE:
            vy_sign = -1 if lateral_offset > 0 else 1
            correction_speed = VY_SHIFT * LATERAL_SPEED_SCALE
            correction_time = abs(lateral_offset) / correction_speed
            mover.move(vx=0, vy=vy_sign * VY_SHIFT, last_time=correction_time)
            lateral_offset += vy_sign * correction_speed * correction_time
            print(f"[归正] 第一次修正后估算横向偏移={lateral_offset:.3f}m")
            if abs(lateral_offset) > TOLERANCE:
                vy_sign = -1 if lateral_offset > 0 else 1
                correction_time = abs(lateral_offset) / correction_speed
                mover.move(vx=0, vy=vy_sign * VY_SHIFT, last_time=correction_time)
                lateral_offset += vy_sign * correction_speed * correction_time
                print(f"[归正] 第二次修正后估算横向偏移={lateral_offset:.3f}m")
        else:
            print("[归正] 已在中心，无需修正")

        print("[任务] 向右转90度...")
        mover.revolve_90()
        print("[任务] 转向完成")

    except KeyboardInterrupt:
        print("[用户] 手动中断")
    finally:
        print(f"[结束] 最终前向={forward_distance:.3f}m, 横向偏移={lateral_offset:.3f}m")
        mover.stop(); time.sleep(0.5)
        cam.stop(); cv2.destroyAllWindows()
        print("[结束] 程序退出")

if __name__ == "__main__":
    main()
