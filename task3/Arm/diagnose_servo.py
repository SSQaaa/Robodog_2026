# -*- coding: utf-8 -*-
"""Read useful SCServo/ST3215 registers for servo troubleshooting."""

import argparse

from config import DEFAULT_CONFIG_PATH, load_config
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import (
    SMS_STS_ACC,
    SMS_STS_GOAL_POSITION_L,
    SMS_STS_GOAL_SPEED_L,
    SMS_STS_LOCK,
    SMS_STS_MIN_ANGLE_LIMIT_L,
    SMS_STS_MODE,
    SMS_STS_OFS_L,
    SMS_STS_PRESENT_CURRENT_L,
    SMS_STS_PRESENT_POSITION_L,
    SMS_STS_PRESENT_TEMPERATURE,
    SMS_STS_PRESENT_VOLTAGE,
    SMS_STS_TORQUE_ENABLE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Read diagnostic registers for one servo.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to params.json.")
    parser.add_argument("--id", type=int, default=5, help="Servo ID to diagnose.")
    return parser.parse_args()


def check(packet, label, result, error):
    if result != COMM_SUCCESS or error:
        return f"{label}=ERR(result={result}, error={error})"
    return None


def read1(packet, servo_id, addr, label):
    value, result, error = packet.read1ByteTxRx(servo_id, addr)
    err = check(packet, label, result, error)
    return err if err else f"{label}={int(value)}"


def read2(packet, servo_id, addr, label, signed=False):
    value, result, error = packet.read2ByteTxRx(servo_id, addr)
    err = check(packet, label, result, error)
    if err:
        return err
    value = int(packet.scs_tohost(value, 15)) if signed else int(value)
    return f"{label}={value}"


def main():
    args = parse_args()
    config = load_config(args.config)
    arm_cfg = config["arm"]
    servo_id = int(args.id)

    port = PortHandler(arm_cfg["devicename"])
    packet = sms_sts(port)
    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {arm_cfg['devicename']}")
    if not port.setBaudRate(int(arm_cfg["baudrate"])):
        port.closePort()
        raise RuntimeError(f"Failed to set servo baudrate: {arm_cfg['baudrate']}")

    try:
        print(f"[Diagnose] opened {arm_cfg['devicename']} @ {arm_cfg['baudrate']}")
        print(f"[Diagnose] id={servo_id}")
        print(read1(packet, servo_id, SMS_STS_MODE, "mode"))
        print(read1(packet, servo_id, SMS_STS_LOCK, "eprom_lock"))
        print(read1(packet, servo_id, SMS_STS_TORQUE_ENABLE, "torque"))
        print(read1(packet, servo_id, SMS_STS_ACC, "goal_acc"))
        print(read2(packet, servo_id, SMS_STS_MIN_ANGLE_LIMIT_L, "internal_min"))
        print(read2(packet, servo_id, SMS_STS_MIN_ANGLE_LIMIT_L + 2, "internal_max"))
        print(read2(packet, servo_id, SMS_STS_OFS_L, "offset", signed=True))
        print(read2(packet, servo_id, SMS_STS_GOAL_POSITION_L, "goal_pos"))
        print(read2(packet, servo_id, SMS_STS_GOAL_SPEED_L, "goal_speed"))
        print(read2(packet, servo_id, SMS_STS_PRESENT_POSITION_L, "present_pos", signed=True))
        print(read2(packet, servo_id, SMS_STS_PRESENT_CURRENT_L, "present_current", signed=True))
        print(read1(packet, servo_id, SMS_STS_PRESENT_VOLTAGE, "voltage"))
        print(read1(packet, servo_id, SMS_STS_PRESENT_TEMPERATURE, "temperature"))

        cfg = arm_cfg.get("servos", {}).get(str(servo_id), {})
        if cfg:
            print(
                "[Config] software_min={} software_max={} zero={} direction={}".format(
                    cfg.get("min"),
                    cfg.get("max"),
                    cfg.get("zero"),
                    cfg.get("direction"),
                )
            )
        reset_pose = arm_cfg.get("reset_pose", {}).get(str(servo_id))
        if reset_pose is not None:
            print(f"[Config] reset_pose={reset_pose}")
    finally:
        port.closePort()
        print("[Diagnose] port closed")


if __name__ == "__main__":
    main()
