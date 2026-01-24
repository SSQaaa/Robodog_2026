from dog_control_sxh_test import DogControl
import time

def Process_Three(robot: DogControl):
    robot.EXmove(vx=20000, last_time=1.1, duration=1, case=1)
    robot.move(vy=50000, last_time=1, duration=1)
    robot.move(vy=-50000, last_time=3.35, duration=0.5)
    robot.start_continue()
    robot.move(vx=8000, last_time=0.15, duration=1)
    robot.close_continue()
