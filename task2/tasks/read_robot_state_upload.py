# -*- coding: utf-8 -*-
"""
只读取机器人状态上传包，不包含任何运动学控制。

用途：
1. 监听 UDP 状态包
2. 筛选 0x0901 RobotStateUpload
3. 打印 roll / pitch / yaw / yaw_vel，方便标定 task2_3 的目标角度
"""

import argparse
import ctypes
import socket
import time


STATE_CODE = 0x0901
HEAD_SIZE = 12


class RobotStateUpload(ctypes.Structure):
    _fields_ = [
        ("robot_basic_state", ctypes.c_int),
        ("robot_gait_state", ctypes.c_int),
        ("robot_policy_state", ctypes.c_int),
        ("rpy", ctypes.c_double * 3),
        ("rpy_vel", ctypes.c_double * 3),
        ("xyz_acc", ctypes.c_double * 3),
        ("pos_world", ctypes.c_double * 3),
        ("vel_world", ctypes.c_double * 3),
        ("vel_body", ctypes.c_double * 3),
        ("touch_down_and_stair_trot", ctypes.c_uint),
        ("is_charging", ctypes.c_bool),
        ("error_state", ctypes.c_uint),
        ("robot_motion_state", ctypes.c_int),
        ("battery_level", ctypes.c_double),
        ("task_state", ctypes.c_int),
        ("is_robot_need_move", ctypes.c_bool),
        ("zero_position_flag", ctypes.c_bool),
        ("is_after_first_start", ctypes.c_bool),
        ("is_voice_ctrl_enable", ctypes.c_bool),
        ("ultrasound", ctypes.c_double * 2),
    ]


class RobotStateUploadPacked(ctypes.Structure):
    _pack_ = 1
    _fields_ = RobotStateUpload._fields_


def parse_command_head(data):
    code = int.from_bytes(data[0:4], byteorder="little", signed=False)
    param_size = int.from_bytes(data[4:8], byteorder="little", signed=False)
    cmd_type = int.from_bytes(data[8:12], byteorder="little", signed=False)
    return code, param_size, cmd_type


def parse_robot_state(payload, packed=False):
    state_cls = RobotStateUploadPacked if packed else RobotStateUpload
    state_size = ctypes.sizeof(state_cls)
    if len(payload) < state_size:
        return None, state_size
    state = state_cls.from_buffer_copy(payload[:state_size])
    return state, state_size


def main():
    parser = argparse.ArgumentParser(description="读取 0x0901 机器人状态上传包")
    parser.add_argument("--host", default="0.0.0.0", help="本机监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, required=True, help="本机监听 UDP 端口")
    parser.add_argument("--interval", type=float, default=0.10, help="打印间隔，单位秒")
    parser.add_argument("--packed", action="store_true", help="如果官方结构体使用 #pragma pack(1)，加这个参数")
    parser.add_argument("--show-all-code", action="store_true", help="打印收到的非 0x0901 指令码")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)

    state_size = ctypes.sizeof(RobotStateUploadPacked if args.packed else RobotStateUpload)
    print("开始监听 UDP {}:{}，等待 0x0901 RobotStateUpload".format(args.host, args.port))
    print("当前解析模式：{}，RobotStateUpload大小={}字节".format("pack(1)" if args.packed else "默认对齐", state_size))
    print("按 Ctrl+C 退出")

    last_print_time = 0.0

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            print("等待状态包中...")
            continue

        if len(data) < HEAD_SIZE:
            continue

        code, param_size, cmd_type = parse_command_head(data)

        if code != STATE_CODE:
            if args.show_all_code:
                print("收到其他指令 code=0x{:04X} type={} size={} from={}".format(code, cmd_type, param_size, addr))
            continue

        if cmd_type != 1:
            print("收到 0x0901，但 type={}，不是复杂指令".format(cmd_type))
            continue

        payload = data[HEAD_SIZE:HEAD_SIZE + param_size]
        state, need_size = parse_robot_state(payload, packed=args.packed)
        if state is None:
            print("收到 0x0901，但正文长度不够：payload={}，解析需要={}".format(len(payload), need_size))
            continue

        now = time.time()
        if now - last_print_time < args.interval:
            continue
        last_print_time = now

        roll = float(state.rpy[0])
        pitch = float(state.rpy[1])
        yaw = float(state.rpy[2])
        roll_vel = float(state.rpy_vel[0])
        pitch_vel = float(state.rpy_vel[1])
        yaw_vel = float(state.rpy_vel[2])

        print(
            "from={} payload={} | roll={:.3f} pitch={:.3f} yaw={:.3f} | "
            "roll_vel={:.4f} pitch_vel={:.4f} yaw_vel={:.4f} | battery={:.2f}".format(
                addr,
                param_size,
                roll,
                pitch,
                yaw,
                roll_vel,
                pitch_vel,
                yaw_vel,
                float(state.battery_level),
            )
        )


if __name__ == "__main__":
    main()
