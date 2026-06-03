# -*- coding: utf-8 -*-
"""Change one SCServo/ST3215 servo ID and baudrate.

Use this with only the target servo connected, otherwise a duplicated source ID
can make the write go to the wrong servo or fail unpredictably.
"""

import argparse
import time

from config import DEFAULT_CONFIG_PATH, load_config
from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
from scservo_sdk.sms_sts import SMS_STS_BAUD_RATE, SMS_STS_ID


BAUD_TO_CODE = {
    1000000: 0,
    500000: 1,
    250000: 2,
    128000: 3,
    115200: 4,
    76800: 5,
    57600: 6,
    38400: 7,
}


def check_comm(packet, action, result, error):
    if result != COMM_SUCCESS or error:
        message = packet.getTxRxResult(result)
        if error:
            message = f"{message}; {packet.getRxPacketError(error)}"
        raise RuntimeError(f"{action} failed: result={result}, error={error}, {message}")


def read_pos(packet, servo_id):
    pos, result, error = packet.ReadPos(int(servo_id))
    check_comm(packet, f"read id={servo_id}", result, error)
    return int(pos)


def open_packet(port_name, baudrate):
    port = PortHandler(port_name)
    packet = sms_sts(port)
    if not port.openPort():
        raise RuntimeError(f"Failed to open servo port: {port_name}")
    if not port.setBaudRate(int(baudrate)):
        port.closePort()
        raise RuntimeError(f"Failed to set baudrate: {baudrate}")
    return port, packet


def main():
    parser = argparse.ArgumentParser(description="Change one SCServo/ST3215 servo ID and baudrate.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to params.json.")
    parser.add_argument("--port", default=None, help="Serial port. Defaults to arm.devicename in params.json.")
    parser.add_argument("--old-id", type=int, required=True, help="Current servo ID.")
    parser.add_argument("--old-baudrate", type=int, required=True, help="Current servo baudrate.")
    parser.add_argument("--new-id", type=int, required=True, help="New servo ID.")
    parser.add_argument("--new-baudrate", type=int, required=True, choices=sorted(BAUD_TO_CODE), help="New baudrate.")
    parser.add_argument("--yes", action="store_true", help="Actually write the servo EPROM.")
    args = parser.parse_args()

    if not 0 <= args.old_id <= 252 or not 0 <= args.new_id <= 252:
        raise ValueError("Servo IDs must be in range 0..252.")
    if args.old_baudrate not in BAUD_TO_CODE:
        raise ValueError(f"Unsupported old baudrate: {args.old_baudrate}")

    config = load_config(args.config)
    port_name = args.port or config["arm"]["devicename"]
    new_baud_code = BAUD_TO_CODE[args.new_baudrate]

    print("[WARNING] Connect only the target servo before running this script.")
    print(
        f"[Plan] port={port_name} id {args.old_id}->{args.new_id}, "
        f"baudrate {args.old_baudrate}->{args.new_baudrate}"
    )
    if not args.yes:
        print("[DryRun] Add --yes to write these settings.")
        return

    port, packet = open_packet(port_name, args.old_baudrate)
    try:
        old_pos = read_pos(packet, args.old_id)
        print(f"[Check] current id={args.old_id} pos={old_pos}")

        result, error = packet.unLockEprom(args.old_id)
        check_comm(packet, "unlock EPROM", result, error)
        time.sleep(0.05)

        result, error = packet.write1ByteTxRx(args.old_id, SMS_STS_ID, int(args.new_id))
        check_comm(packet, f"write new ID {args.new_id}", result, error)
        time.sleep(0.05)

        write_id = args.new_id
        result, error = packet.write1ByteTxRx(write_id, SMS_STS_BAUD_RATE, int(new_baud_code))
        check_comm(packet, f"write new baudrate code {new_baud_code}", result, error)
        time.sleep(0.05)

        result, error = packet.LockEprom(write_id)
        if result != COMM_SUCCESS or error:
            print("[Info] Servo stopped replying on the old baudrate after baudrate write.")
            print("[Info] Reopening the port with the new baudrate to lock EPROM.")
            port.closePort()
            port, packet = open_packet(port_name, args.new_baudrate)
            read_pos(packet, args.new_id)
            result, error = packet.LockEprom(args.new_id)
        check_comm(packet, "lock EPROM", result, error)
    finally:
        port.closePort()

    print("[Done] Settings written.")
    print("[Next] Power-cycle the servo, then verify with:")
    print(f"  python Arm/scan_servo_ids.py --ids {args.new_id} --baudrate {args.new_baudrate}")


if __name__ == "__main__":
    main()
