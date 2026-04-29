# -*- coding: utf-8 -*-
import cv2
import numpy as np
import orbbec_native
import time

cam = orbbec_native.OrbbecCamera()
cam.start()
time.sleep(1)

color_w, color_h = cam.get_color_size()
print(f"Image Size: {color_w}x{color_h}")
print(f"Center Point: ({color_w//2}, {color_h//2})")

print("\nInstructions:")
print("1. Place the object at the center of the image and remember this position.")
print("2. Move the object 50mm to the right using a ruler.")
print("3. Observe how many pixels the object has moved in the image.")
print("4. Press ESC to exit.\n")

while True:
    frame = cam.get_color_frame()
    if frame is not None:
        frame = np.asarray(frame, dtype=np.uint8)

        # 画中心线
        cv2.line(frame, (color_w//2, 0), (color_w//2, color_h), (0,255,0), 1)
        cv2.line(frame, (0, color_h//2), (color_w, color_h//2), (0,255,0), 1)

        # 刻度线：每10像素一根（密集）
        for i in range(0, color_w, 10):
            cv2.line(frame, (i, color_h//2-10), (i, color_h//2+10), (255,0,0), 1)
        
        # 刻度值：每50像素显示一个数字（清晰）
        for i in range(0, color_w, 50):
            cv2.putText(frame, str(i), (i-15, color_h//2+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

        cv2.imshow("Calibrate", frame)

    if cv2.waitKey(1) == 27:
        break

cam.stop()
cv2.destroyAllWindows()