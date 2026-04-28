import cv2

# 打开默认摄像头（索引通常为0）
cap = cv2.VideoCapture(2)

# 设置分辨率为1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查是否设置成功
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"实际分辨率: {actual_width}x{actual_height}")

# 读取并显示画面
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    print(frame.shape)
    cv2.imshow('Camera', frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()