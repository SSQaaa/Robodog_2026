# task3_new

独立的绿色物块视觉抓取代码。旧 `task3` 保留不动。

## 常用命令

```bash
python disable_servos.py
python disable_servos.py --enable --ids 3 4 5
python monitor_servos.py --ids 3 4 5 --single-line
python monitor_servos.py --ids 3 4 5 --warn-current 80 --high-current 150 --danger-current 250
python reset_arm.py
python reset_arm.py --open-gripper
python kinematics.py --x 450 --y 0 --z 25
python kinematics.py --theta1 46.27 --theta2 25.10 --theta3 18.63
python vision_grasp.py --dry-run
python calibration.py --samples 60 --save
python vision_grasp.py --show
python vision_grasp.py --execute
```

## 文件说明

- `params.json`: 全部参数入口，包括舵机、连杆、夹爪限流、绿色阈值、相机安全球和标定矩阵。
- `config.py`: 读取和保存参数文件。
- `servo_driver.py`: 舵机底层封装，包含写位置、读状态、扭矩开关和夹爪限流闭合。
- `disable_servos.py`: 舵机失能/使能。
- `monitor_servos.py`: 实时打印舵机位置、电流、温度和状态分级 `OK/WARN/HIGH/DANGER`。
- `reset_arm.py`: 按 `params.json` 里的 `arm.reset_pose` 回到初始姿态。
- `camera_safety.py`: 相机安全球危险判断，IK 筛选候选解时调用。
- `kinematics.py`: 三连杆逆解和姿态显示，默认会尝试读取并打印当前舵机 pos，并自动显示连杆姿态图。
- `calibration.py`: AprilTag 检测、坐标变换和 `T_base_camera` 标定。
- `vision_grasp.py`: 直接调用 `orbbec_native`，使用 `params.json` 里保存的 `camera.T_base_camera` 完成绿色物块识别、深度取点、逆解和抓取流程；抓取时不再实时检测 AprilTag。
- `scservo_sdk/`: ST3215/SCServo SDK。

## 角度约定

`theta1/theta2/theta3` 都是相对关节角：

- `theta1=0`: 连杆1垂直地面。
- `theta2=0`: 连杆2和连杆1同一直线。
- `theta3=0`: 连杆3和连杆2同一直线。
- 抓手水平时：`theta1 + theta2 + theta3 = 90 deg`。

servo5 对应 `theta1`，servo4 对应 `theta2`，servo3 对应 `theta3`。

## 相机安全球

相机安全球参数在 `params.json` 的 `arm.camera_safety`：

```json
"camera_safety": {
  "enabled": true,
  "center_base_mm": [150.0, 0.0, 80.0],
  "radius_mm": 50.0
}
```

IK 会把相机安全球投影到当前底座 yaw 的机械臂平面，检查 `L1/L2/L3/gripper` 每段线段到安全球的距离。若任意线段进入安全半径，该候选解会被丢弃。

## 标定和抓取

相机固定后，只需要先标定一次：

```bash
python calibration.py --samples 60 --save
```

标定会把 `T_base_camera` 保存到 `params.json`。之后抓取程序只读取这个保存值，不会再实时检测 AprilTag：

```bash
python vision_grasp.py --dry-run
python vision_grasp.py --execute
```

## 舵机底层限位

`params.json` 里的 `min/max` 是代码层的软件限位；舵机内部 EEPROM 里也有底层角度限位。ST3215/SCServo 常用地址：

- `9`: 最小角度限位低字节地址，读 2 字节。
- `11`: 最大角度限位低字节地址，读 2 字节。
- `56`: 当前角度位置低字节地址，通常直接用 `ReadPos(id)`。

查看某个舵机的底层限位和当前位置，例如查看 5 号舵机：

```bash
python -c "from scservo_sdk import PortHandler, sms_sts; port=PortHandler('/dev/ttyUSB0'); p=sms_sts(port); port.openPort(); port.setBaudRate(500000); print('5 min', p.read2ByteTxRx(5,9)); print('5 max', p.read2ByteTxRx(5,11)); print('5 pos', p.ReadPos(5)); port.closePort()"
```

修改底层限位前先确认机械臂处于安全姿态，并建议先失能：

```bash
python disable_servos.py
```

修改 5 号舵机底层限位示例，假设要设成 `min=300, max=1700`：

```bash
python -c "from scservo_sdk import PortHandler, sms_sts; port=PortHandler('/dev/ttyUSB0'); p=sms_sts(port); port.openPort(); port.setBaudRate(500000); print('unlock', p.unLockEprom(5)); print('write min', p.write2ByteTxRx(5,9,300)); print('write max', p.write2ByteTxRx(5,11,1700)); print('lock', p.LockEprom(5)); print('5 min', p.read2ByteTxRx(5,9)); print('5 max', p.read2ByteTxRx(5,11)); port.closePort()"
```

修改后再用监控确认当前位置和运动范围：

```bash
python monitor_servos.py --ids 5 --single-line
```

注意：底层限位会写入舵机自身，影响所有程序；`params.json` 只影响本项目代码。两边建议保持一致或让 `params.json` 更保守。
