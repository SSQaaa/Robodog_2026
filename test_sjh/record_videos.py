# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import cv2
import time
import os
from datetime import datetime

# 参数设置
DEV = '/dev/video0'
OUTDIR = '/mnt/uu/videos'
FPS = 25
SEGSEC = 300  # 每段秒数

os.makedirs(OUTDIR, exist_ok=True)

cap = cv2.VideoCapture(DEV)
if not cap.isOpened():
    print("camera failed")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0
start_time = time.time()
out = None

print(f"Recording... DEV={DEV} OUTDIR={OUTDIR} FPS={FPS} SEGSEC={SEGSEC} (Ctrl+C stop)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("camera failed")
            break

        # 如果当前段视频不存在或时间到达新段
        if out is None or time.time() - start_time >= SEGSEC:
            if out:
                out.release()
            TS = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTDIR, f"cam_{TS}_{i:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(filename, fourcc, FPS, (width, height))
            print(f"new video: {filename}")
            start_time = time.time()
            i += 1

        # 写入视频
        out.write(frame)

        # 显示画面
        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("exit")

finally:
    if out:
        out.release()
    cap.release()
    cv2.destroyAllWindows()