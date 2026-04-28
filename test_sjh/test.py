# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from dog_control import DogControl

def color_detect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    frame_cnt = 0
    fps = 0
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera Error")
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([5, 100, 100])
        upper_blue = np.array([15, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours[-2] 
     
        if cnts:
          max_cnt = max(cnts, key=cv2.contourArea)
          if cv2.contourArea(max_cnt) > 500:
              x, y, w, h = cv2.boundingRect(max_cnt)
              print(x+w//2, y+h//2, "Area:", w*h)
              cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        frame_cnt += 1
        
        if frame_cnt == 5:
            end_time = time.time()
            fps = 1/5/(end_time - start_time)
            frame_cnt = 0
        
        cv2.putText(frame, f"FPS: {int(fps)}", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    control = DogControl()
    time.sleep(1)
    control.close_continue()
    control.stop()
    time.sleep(0.5)
    control.stand_up()
    time.sleep(0.5)   
    
    color_detect()
    
