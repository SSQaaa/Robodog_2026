import orbbec_native
import numpy as np
import cv2

cam = orbbec_native.OrbbecCamera()
cam.start()
time.sleep(1.0)

depth_w, depth_h = cam.get_depth_size()
print(f"Depth size: {depth_w}x{depth_h}")

while True:
    # 直接获取整张深度图（如果你的绑定没有暴露整体深度图，可以看下面备注）
    color = cam.get_color_frame()
    depth = cam.get_depth_frame()  # 如果 OrbbecCamera 没有这个方法，说明封装时没提供
    
    # 如果无法获取整张深度图，就调用 get_depth_in_box 测试中心区域
    center_x, center_y = depth_w // 2, depth_h // 2
    d, v = cam.get_depth_in_box(center_x-10, center_y-10, center_x+10, center_y+10)
    print(f"Center depth: {d} mm, valid: {v}")
    
    if v > 0:
        break

cam.stop()