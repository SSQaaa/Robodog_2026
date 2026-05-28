# -*- coding: utf-8 -*-
"""任务三使用的 ST3215/SCServo 舵机控制封装。"""

import time
from dataclasses import dataclass

from config import all_servo_ids, clamp, servo_cfg
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import (
    SMS_STS_PRESENT_CURRENT_L,
    SMS_STS_PRESENT_TEMPERATURE,
    SMS_STS_PRESENT_VOLTAGE,
    SMS_STS_TORQUE_ENABLE,
)


@dataclass
class ServoStatus:
    servo_id: int
    position: int = None
    current_units: int = None
    voltage_raw: int = None
    temperature_c: int = None
    error: str = None


class ServoBus:
    def __init__(self, arm_cfg):
        self.cfg = arm_cfg
        self.ids = arm_cfg["ids"]
        self.speed = int(arm_cfg["moving_speed"])
        self.acc = int(arm_cfg["moving_acc"])
        self.current_ma_per_unit = float(arm_cfg["gripper"].get("current_ma_per_unit", 6.5))
        self.port = PortHandler(arm_cfg["devicename"])
        self.packet = sms_sts(self.port)
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open servo port: {arm_cfg['devicename']}")
        if not self.port.setBaudRate(int(arm_cfg["baudrate"])):
            self.port.closePort()
            raise RuntimeError(f"Failed to set servo baudrate: {arm_cfg['baudrate']}")
        print(f"[ServoBus] opened {arm_cfg['devicename']} @ {arm_cfg['baudrate']}")

    def close(self):
        self.port.closePort()
        print("[ServoBus] port closed")

    def angle_to_pos(self, servo_id, angle_deg):
        cfg = self.cfg.get("servos", {}).get(str(int(servo_id)), {})
        zero = float(cfg.get("zero", 2047))
        direction = float(cfg.get("direction", 1))
        ticks_per_degree = float(self.cfg.get("ticks_per_degree", 4096.0 / 360.0))
        raw = zero + direction * float(angle_deg) * ticks_per_degree
        return int(round(clamp(raw, int(cfg.get("min", 0)), int(cfg.get("max", 4095)))))

    def check_limit(self, servo_id, position):
        cfg = self.cfg.get("servos", {}).get(str(int(servo_id)), {})
        low = int(cfg.get("min", 0))
        high = int(cfg.get("max", 4095))
        if int(position) < low or int(position) > high:
            raise ValueError(f"Servo {servo_id} target {position} outside limit [{low}, {high}]")

    def write_pos(self, servo_id, position, speed=None, acc=None):
        self.check_limit(servo_id, position)
        speed = self.speed if speed is None else int(speed)
        acc = self.acc if acc is None else int(acc)
        print(f"[ServoBus] target id={servo_id} pos={int(position)} speed={speed} acc={acc}")
        result, error = self.packet.WritePosEx(int(servo_id), int(position), speed, acc)
        if result != COMM_SUCCESS:
            print(self.packet.getTxRxResult(result))
        if error:
            print(self.packet.getRxPacketError(error))
        return result, error

    def move_targets(self, targets, wait_s=1.0, read_after=True):
        for servo_id, position in targets.items():
            self.write_pos(int(servo_id), int(position))
            time.sleep(0.04)
        time.sleep(float(wait_s))
        if read_after:
            self.print_status(sorted(int(sid) for sid in targets.keys()))

    def set_torque(self, servo_id, enabled):
        value = 1 if enabled else 0
        result, error = self.packet.write1ByteTxRx(int(servo_id), SMS_STS_TORQUE_ENABLE, value)
        print(f"[ServoBus] torque id={servo_id} enabled={enabled} result={result} error={error}")
        return result, error

    def set_torque_many(self, servo_ids, enabled):
        for servo_id in servo_ids:
            self.set_torque(int(servo_id), enabled)
            time.sleep(0.04)

    def read_status(self, servo_id):
        servo_id = int(servo_id)
        status = ServoStatus(servo_id=servo_id)
        position, result, error = self.packet.ReadPos(servo_id)
        if result != COMM_SUCCESS or error:
            status.error = f"pos_result={result}, pos_error={error}"
            return status
        status.position = int(position)

        raw_current, result, error = self.packet.read2ByteTxRx(servo_id, SMS_STS_PRESENT_CURRENT_L)
        if result == COMM_SUCCESS and not error:
            status.current_units = int(self.packet.scs_tohost(raw_current, 15))

        voltage, result, error = self.packet.read1ByteTxRx(servo_id, SMS_STS_PRESENT_VOLTAGE)
        if result == COMM_SUCCESS and not error:
            status.voltage_raw = int(voltage)

        temperature, result, error = self.packet.read1ByteTxRx(servo_id, SMS_STS_PRESENT_TEMPERATURE)
        if result == COMM_SUCCESS and not error:
            status.temperature_c = int(temperature)

        return status

    def print_status(self, servo_ids=None):
        servo_ids = servo_ids or all_servo_ids({"arm": self.cfg})
        for servo_id in servo_ids:
            status = self.read_status(servo_id)
            if status.error:
                print(f"[ServoStatus] id={servo_id} ERR {status.error}")
                continue
            ma = None
            if status.current_units is not None:
                ma = status.current_units * self.current_ma_per_unit
            print(
                "[ServoStatus] id={} pos={} current={}({}) voltage={} temp={}".format(
                    servo_id,
                    status.position,
                    status.current_units,
                    "n/a" if ma is None else f"{ma:.1f}mA",
                    status.voltage_raw,
                    status.temperature_c,
                )
            )

    def open_gripper(self):
        self.write_pos(self.ids["gripper"], self.cfg["gripper"]["open"])
        time.sleep(0.5)
        self.print_status([self.ids["gripper"]])

    def close_gripper_protected(self):
        cfg = self.cfg["gripper"]
        servo_id = int(self.ids["gripper"])
        target = int(cfg["close"])
        limit = int(cfg.get("current_limit_units", 80))
        step = max(1, int(cfg.get("close_step_ticks", 20)))
        retreat = max(0, int(cfg.get("retreat_ticks", 30)))
        interval = max(0.02, float(cfg.get("sample_interval_s", 0.05)))
        deadline = time.time() + float(cfg.get("max_close_s", 2.0))

        status = self.read_status(servo_id)
        start = status.position if status.position is not None else int(cfg["open"])
        direction = 1 if target >= start else -1
        pos = start
        print(f"[Gripper] protected close start={start} target={target} current_limit={limit}")
        while time.time() < deadline:
            if (direction > 0 and pos >= target) or (direction < 0 and pos <= target):
                break
            pos += direction * step
            if direction > 0:
                pos = min(pos, target)
            else:
                pos = max(pos, target)
            self.write_pos(servo_id, pos)
            time.sleep(interval)
            status = self.read_status(servo_id)
            self.print_status([servo_id])
            if status.current_units is not None and abs(status.current_units) >= limit:
                hold = pos - direction * retreat
                hold = int(clamp(hold, servo_cfg({"arm": self.cfg}, servo_id).get("min", 0), servo_cfg({"arm": self.cfg}, servo_id).get("max", 4095)))
                print(f"[Gripper] current limit reached: {status.current_units}; retreat/hold at {hold}")
                self.write_pos(servo_id, hold)
                return hold
        print(f"[Gripper] close finished at {pos}")
        return pos
