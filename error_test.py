#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
import orbbec_native

cam = orbbec_native.OrbbecCamera()
cam.start()
time.sleep(1.0)

depth_w, depth_h = cam.get_depth_size()
color_w, color_h = cam.get_color_size()
print(f"Depth: {depth_w}x{depth_h}, Color: {color_w}x{color_h}")

# 测试中心点深度
u = depth_w // 2
v = depth_h // 2
d, valid = cam.get_depth_in_box(u-20, v-20, u+20, v+20)
print(f"Center depth: {d} mm, valid: {valid}")

# 测试左上角
d2, v2 = cam.get_depth_in_box(0, 0, 50, 50)
print(f"Top-left depth: {d2} mm, valid: {v2}")

# 如果可能，尝试直接获取深度图中的单个像素（如果你的绑定没有此函数，请忽略）
try:
    d_point = cam.get_depth_at(u, v)   # 如果实现了
    print(f"Single pixel depth: {d_point} mm")
except AttributeError:
    print("get_depth_at not available")

cam.stop()