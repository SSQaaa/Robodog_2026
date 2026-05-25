# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics competition project for a Unitree Go1 robot dog with a 6-DOF mechanical arm. The project implements Task3: a block transportation task where the robot detects colored blocks (Red/Green) and places them into designated boxes (A/B/C/D) based on dashboard status readings.

## Repository Structure

```
2026Project/
├── task3/                    # Main task implementation
│   ├── task3.py             # Mission orchestration entry point
│   ├── vision_control.py    # TensorRT YOLO + Orbbec depth camera
│   ├── arm_control.py       # Arm control interface
│   ├── dog_control.py       # Robot dog UDP movement control
│   ├── udp.py               # UDP client for dog communication
│   ├── obj_det.py           # Standalone YOLO detection validation script
│   └── Arm/                 # Arm-specific code and configuration
│       ├── params.json      # Arm kinematics, calibration, servo limits
│       ├── kinematics.py    # 3-DOF planar IK solver
│       ├── servo_driver.py  # ST3215/SCServo舵机驱动封装
│       ├── vision_grasp.py  # Green block detection + grasp execution
│       ├── calibration.py   # Camera-to-arm calibration using AprilTag
│       ├── camera_safety.py # Camera collision avoidance
│       ├── reset_arm.py     # Reset arm to default position
│       ├── monitor_servos.py# Real-time servo monitoring
│       └── disable_servos.py# Enable/disable servo torque
├── task1/                   # Previous task implementations
├── task2/
└── libs/                    # TensorRT libraries and YOLO engine files
```

## Key Components

### Vision System (`vision_control.py`)
- Uses TensorRT YOLO model (`bigdog_0427.engine`) for object detection
- Orbbec depth camera for distance measurement
- Detected classes: A, B, C, D, Green, Red, MPa, Traffic_cone, dashboard, ssi
- Depth smoothing with median filter over 5 frames

### Arm System (`Arm/`)
- 6-DOF arm with 3 active joints for planar IK (shoulder, elbow, wrist)
- Coordinate frames: camera -> base (via `T_base_camera` calibration matrix)
- Key config in `params.json`:
  - `camera.T_base_camera`: 4x4 calibration matrix (run `calibration.py --samples 60 --save` to generate)
  - `arm.link_lengths_mm`: L1=150, L2=145, L3=80
  - `arm.servos`: Zero positions, directions, and limits per servo
  - `block`: Size, grasp offsets, table height

### Dog Control (`dog_control.py`)
- UDP communication to dog at `192.168.1.120:43893`
- Commands: heartbeat, stand_up, move_mode, velocity (vx/vy/vz), actions (turn 180°)

## Common Commands

### Run the Main Task
```bash
cd 2026Project/task3
python task3.py
```

### Validate Vision Only
```bash
cd 2026Project/task3
python obj_det.py
```

### Arm Calibration and Testing (in `Arm/` directory)
```bash
cd Arm

# Camera-to-arm calibration (uses AprilTag id=0)
python calibration.py --samples 60 --save

# Reset arm to default position
python reset_arm.py
python reset_arm.py --open-gripper

# Monitor servo status
python monitor_servos.py --ids 3 4 5 --single-line

# Enable/disable servo torque
python disable_servos.py
python disable_servos.py --enable --ids 3 4 5

# Test grasp (dry run first)
python vision_grasp.py --dry-run
python vision_grasp.py --execute

# Test IK solver
python kinematics.py --x 300 --y 0 --z 200
python kinematics.py --theta1 0 --theta2 45 --theta3 45  # Plot specific pose
```

### Debug IK Without Hardware
```python
# In Python REPL or script
from Arm.kinematics import solve_arm_target, print_solution
from Arm.config import load_config

config = load_config("Arm/params.json")
solution = solve_arm_target(x=300, y=0, z=200, arm_cfg=config["arm"])
print_solution(solution)
```

## Configuration Constants (in `task3.py`)

Key tunables for competition:
```python
DASHBOARD_STATUS = {"A": "ABNORMAL", "B": "NORMAL", "C": "ABNORMAL", "D": "NORMAL"}
# ABNORMAL -> pick Red, NORMAL -> pick Green

DRY_RUN = False  # Set True to skip actual servo movement
CENTER_TOLERANCE_BLOCK_PX = 100  # Pixel tolerance for block alignment
CENTER_TOLERANCE_BOX_PX = 20     # Pixel tolerance for box alignment
GRASP_R_LIMIT_MM = 430.0         # Max reachable distance for grasp
FORWARD_SECONDS = 2.0            # Forward movement after 180° turn
FINAL_BOX_FORWARD_SECONDS = 2.5  # Final approach to box
RETURN_FORWARD_SECONDS = 5.0     # Return to pickup zone
```

## Architecture Patterns

1. **Task Orchestration**: `task3.py` coordinates high-level mission flow but delegates to specialized controllers:
   - `VisionControl` for detection
   - `ArmControl` for grasping (uses `Arm/` modules)
   - `DogControl` for movement

2. **Dry-Run Mode**: All arm operations support `dry_run` flag for testing logic without hardware.

3. **Calibration Workflow**:
   - `camera.T_base_tag` defines tag position in base frame (fixed mount)
   - `calibration.py` detects AprilTag in camera frame, solves for `T_base_camera`
   - Vision detections use `T_base_camera` to transform to base frame for IK

4. **Safety**: `camera_safety.py` checks if arm pose would collide with camera and rejects unsafe IK solutions.

## Important Notes

- Robot dog uses big-endian UDP commands; arm uses SCServo protocol on `/dev/ttyUSB0`
- TensorRT engine and `orbbec_native` module are pre-built binaries in `libs/`
- Servo IDs: gripper=1, gripper_rotate=2, wrist=3, elbow=4, shoulder=5, base=6
- When modifying arm geometry or adding collision avoidance, update `params.json` and test with `kinematics.py --dry-run` first
