# -*- coding: utf-8 -*-
import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(PROJECT_DIR, "libs")
ENGINE_PATH = os.path.join(LIBS_DIR, "bigdog_0427.engine")
PLUGIN_PATH = os.path.join(LIBS_DIR, "libmyplugins.so")

CLASS_NAMES = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "Green",
    5: "MPa",
    6: "Red",
    7: "Traffic_cone",
    8: "dashboard",
    9: "ssi",
}
LETTER_ID_TO_NAME = {0: "A", 1: "B", 2: "C", 3: "D"}
DASHBOARD_ID = 8
SSI_ID = 9

STATE_CN_MAP = {
    "normal": "正常",
    "low": "偏低",
    "high": "偏高",
    "unknown": "未知",
}
STATUS_CN_TO_TASK3 = {
    "正常": "NORMAL",
    "偏低": "ABNORMAL",
    "偏高": "ABNORMAL",
}
UNKNOWN_STATE_CN = "未知"

################################################################################
# 状态缩写：L=偏低（异常），H=偏高（异常），Z=正常。
# 写死：True，不写死:False
task2stable = True
# False：字母水平/深度对齐后跳过 UPDOWN、SSI 和仪表盘指针识别，
#        等待 2 秒并直接使用 TASK2_DASHBOARD_CONFIG 播报和记录。

# True ：保留原有完整仪表盘识别流程。
# 不蹲：False,蹲 true
task2recognize_dashboard = False
TASK2_DASHBOARD_CONFIG = (
    ("B", "L"),
    ("C", "H"),
    ("D", "Z"),
    ("A", "Z"),
)

DEFAULT_DASHBOARD_STATUS = {
    "A": "NORMAL",
    "B": "ABNORMAL",
    "C": "ABNORMAL",
    "D": "NORMAL",
}


BOX_LETTER_ORDER = ("A", "B", "C", "D")

##############################################################################
DOG_IP = "192.168.1.120"
DOG_PORT = 43893

FORWARD_SPEED = 20000
APPROACH_BOX_SPEED = 10000
BACKWARD_SPEED = -20000
PRE_GRASP_MOVE_SPEED_X = 7000
PRE_GRASP_MOVE_SPEED_Y = 25000
SIDE_SPEED = 25000

ARM_BAUDRATE = 500000
ARM_DEVICE = "/dev/ttyUSB0"
ARM_MOVING_SPEED = 1500
ARM_MOVING_ACC = 50
ARM_CURRENT_THRESHOLD = 30

ARM_RESET_POSE = {
    1: 2400,
    2: 2047,
    3: 3080,
    4: 800,
    5: 2400,
}

ARM_GRIPPER_OPEN_POS = 1600
ARM_GRIPPER_CLOSE_POS = 2400

ARM_ANGLE_LIMITS = {
    3: (1000, 3200),
    4: (540, 3400),
    5: (1000, 3050),
}

TASK1_CONE_CLASS_ID = 7
TASK1_TARGET_FORWARD_M = 4.5
TASK1_CONE_PASS_DEPTH_MIN_MM = 500
TASK1_CONE_PASS_DEPTH_MAX_MM = 800
TASK1_CONE_RELIABLE_DEPTH_MM = 1800
TASK1_DEPTH_INVALID_AS_MM = 2500
TASK1_CONE_LOST_FRAMES = 3
TASK1_MARK_TIMEOUT_S = 8.0
TASK1_MARK_MAX_FRAMES = 40
TASK1_ALIGN_TIMEOUT_S = 300.0
TASK1_ALIGN_VX = 12000
TASK1_ALIGN_VY = 18000
TASK1_ALIGN_STEP_S = 0.12
TASK1_PRE_APPROACH_VX = 10000
TASK1_PRE_APPROACH_STEP_S = 0.25
TASK1_PASS_CONE_VX = 10000
TASK1_PASS_CONE_STEP_S = 0.30
TASK1_FORWARD_VX = 12000
TASK1_FORWARD_STEP_S = 0.18
TASK1_WORLD_POSE_TIMEOUT_S = 8.0
TASK1_WORLD_POSE_PYTHON = "/usr/bin/python2"
