# -*- coding: utf-8 -*-
import sys
import time
sys.path.append("..")
from scservo_sdk import *
from three_Inverse_kinematics import Arm

BAUDRATE = 500000
DEVICENAME = '/dev/ttyUSB0'

# ========== Safety Parameters (Same as write.py) ==========
SCS_MOVING_SPEED = 1500
SCS_MOVING_ACC = 50
CURRENT_THRESHOLD = 30

# Angle Limits
ANGLE3_MIN, ANGLE3_MAX = 1000, 3200
ANGLE4_MIN, ANGLE4_MAX = 540, 3400
ANGLE5_MIN, ANGLE5_MAX = 1000, 3050

class SafeRobotArm:
    def __init__(self):
        self.port = PortHandler(DEVICENAME)
        self.packet = sms_sts(self.port)
        if not self.port.openPort():
            raise Exception("Failed to open serial port")
        if not self.port.setBaudRate(BAUDRATE):
            raise Exception("Failed to set baudrate")
        print("[Robot Arm] Connected")

    def move_to_xy(self, x, y):
        try:
            angle_3, angle_4, angle_5 = Arm(x, y)
            print(f"Calculated Angles: 3={angle_3}, 4={angle_4}, 5={angle_5}")

            if (angle_3 < ANGLE3_MIN or angle_3 > ANGLE3_MAX or
                angle_4 < ANGLE4_MIN or angle_4 > ANGLE4_MAX or
                angle_5 < ANGLE5_MIN or angle_5 > ANGLE5_MAX):
                print(f"Angle out of limit! Skip this position")
                return False

            self.packet.WritePosEx(3, angle_3, SCS_MOVING_SPEED, SCS_MOVING_ACC)
            self.packet.WritePosEx(4, angle_4, SCS_MOVING_SPEED, SCS_MOVING_ACC)
            self.packet.WritePosEx(5, angle_5, SCS_MOVING_SPEED, SCS_MOVING_ACC)

            time.sleep(2)

            print(f"[Robot Arm] Moved to X={x}, Y={y}")
            return True

        except Exception as e:
            print(f"Movement failed: {e}")
            return False

    def gripper_open(self):
        position = 1600
        self.packet.WritePosEx(1, position, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        time.sleep(2)
        print("[Gripper] Opened")

    def gripper_close_with_protection(self):
        target_position = 2400

        self.packet.WritePosEx(1, target_position, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        time.sleep(2)

        print("[Gripper] Starting current detection...")
        while True:
            current, position = self.packet.ReadPosStatus(1)
            print(f"[Gripper] Current={current}, Position={position}")

            self.packet.WritePosEx(1, position, SCS_MOVING_SPEED, SCS_MOVING_ACC)
            time.sleep(0.05)

            if current > CURRENT_THRESHOLD:
                print(f"Over current({current}>{CURRENT_THRESHOLD}), retreating!")
                position = position - 1
                self.packet.WritePosEx(1, position, SCS_MOVING_SPEED, SCS_MOVING_ACC)
            else:
                print(f"Current normal({current}<={CURRENT_THRESHOLD}), holding")
                break

        print("[Gripper] Closed with protection")

    def init_pose(self):
        print("[Robot Arm] Reset to initial pose...")
        self.packet.WritePosEx(1, 2400, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        self.packet.WritePosEx(2, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        self.packet.WritePosEx(3, 3080, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        self.packet.WritePosEx(4, 800, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        self.packet.WritePosEx(5, 2400, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        time.sleep(3)
        print("[Robot Arm] Reset complete")

    def close(self):
        self.port.closePort()

def simple_grasp_demo():
    arm = SafeRobotArm()

    arm.init_pose()

    print("\n========== Safe Grasp Test ==========")
    print("Step 1: Open gripper")
    arm.gripper_open()

    test_x = 150
    test_y = 0

    print(f"\nStep 2: Move to X={test_x}, Y={test_y}")
    input("Press Enter to continue after confirming safety...")

    if arm.move_to_xy(test_x, test_y):
        print(f"\nStep 3: Close gripper (with over-current protection)")
        arm.gripper_close_with_protection()

        print(f"\nStep 4: Lift up")
        arm.move_to_xy(test_x, test_y - 30)

        print(f"\nStep 5: Return to home position")
        arm.init_pose()

    arm.close()
    print("Test completed")

if __name__ == "__main__":
    simple_grasp_demo()