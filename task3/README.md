# Task3 现场运行与舵机维护说明

本目录是任务三主程序，包含视觉识别、机器狗移动、机械臂抓取和放置流程。

## 常用入口

```bash
cd ~/Desktop/2026Project/task3
python task3.py
```

机械臂相关工具在 `Arm/` 目录下，也可以从 `task3` 根目录用 `python Arm/xxx.py` 运行。

## 舵机 ID 与波特率

项目默认使用 `/dev/ttyUSB0`，波特率为 `500000`。配置位置：

```json
"arm": {
  "devicename": "/dev/ttyUSB0",
  "baudrate": 500000
}
```

当前机械臂舵机 ID 约定：

| 关节 | ID |
| --- | --- |
| gripper | 1 |
| gripper_rotate | 2 |
| wrist | 3 |
| elbow | 4 |
| shoulder | 5 |
| base | 6 |

如果状态里出现类似：

```text
id5:ERR pos_result=-6, pos_error=0
```

通常表示程序按 `id=5` 读取舵机时没有收到正常回复。常见原因是新舵机的 ID 不是 5、波特率不是 500000、接线/供电异常，或者协议型号不匹配。

## 扫描舵机 ID

新增脚本：

```text
Arm/scan_servo_ids.py
```

默认使用 `Arm/params.json` 里的串口和波特率，扫描 `0-20`：

```bash
cd ~/Desktop/2026Project/task3
python Arm/scan_servo_ids.py
```

扫描指定范围：

```bash
python Arm/scan_servo_ids.py --ids 1-6
python Arm/scan_servo_ids.py --ids 0-252
```

扫描指定波特率：

```bash
python Arm/scan_servo_ids.py --ids 1-6 --baudrate 500000
python Arm/scan_servo_ids.py --ids 0-20 --baudrate 1000000
```

尝试常见波特率：

```bash
python Arm/scan_servo_ids.py --ids 0-20 --try-common-baudrates
```

目前脚本会依次尝试：

```text
500000, 1000000, 115200, 250000
```

扫描到舵机会输出：

```text
[FOUND] id=5 pos=2033
```

如果整条机械臂都接着扫描，可能会遇到 ID 重复或波特率不同的舵机。为了确认新舵机，最稳的方法是只接新舵机扫描；如果现场不方便单独接，也可以利用不同波特率区分，但写配置时要更小心。

## 修改新舵机 ID 和波特率

新增脚本：

```text
Arm/set_servo_id_baud.py
```

用途：把一个舵机从旧 ID/旧波特率改成项目需要的新 ID/新波特率。

强烈建议只接目标舵机再执行写入操作。如果不能只接目标舵机，至少要确认目标舵机所在波特率下没有其他相同 ID 的舵机，否则可能误写。

先 dry-run，确认计划：

```bash
cd ~/Desktop/2026Project/task3
python Arm/set_servo_id_baud.py --old-id 1 --old-baudrate 1000000 --new-id 5 --new-baudrate 500000
```

确认无误后加 `--yes` 真正写入：

```bash
python Arm/set_servo_id_baud.py --old-id 1 --old-baudrate 1000000 --new-id 5 --new-baudrate 500000 --yes
```

写入后必须给舵机断电重启，然后验证：

```bash
python Arm/scan_servo_ids.py --ids 5 --baudrate 500000
```

整条机械臂接回后再扫：

```bash
python Arm/scan_servo_ids.py --ids 1-6 --baudrate 500000
```

正常应能看到 `1,2,3,4,5,6` 都被 `[FOUND]`。

### 例子：新舵机显示为 1000000/id=1

如果扫描结果类似：

```text
[Scan] port=/dev/ttyUSB0 baudrate=500000 ids=0..20
[FOUND] id=1 pos=2225
[FOUND] id=2 pos=2002
[FOUND] id=3 pos=2061
[FOUND] id=4 pos=2070
[FOUND] id=6 pos=2033
[Scan] port=/dev/ttyUSB0 baudrate=1000000 ids=0..20
[FOUND] id=1 pos=3375
```

这通常表示旧的 `id=1` 在 `500000` 上，新舵机可能是 `id=1`、`baudrate=1000000`。要把它改成肩关节 `id=5`：

```bash
python Arm/set_servo_id_baud.py --old-id 1 --old-baudrate 1000000 --new-id 5 --new-baudrate 500000 --yes
```

如果不能只接新舵机，这个操作大概率只会影响 `1000000/id=1` 的舵机，因为旧 `500000/id=1` 听不懂 1000000 波特率下的命令。但这不是零风险，执行前务必确认没有其他 `1000000/id=1` 的舵机。

## 抓取后的抬升逻辑

`ArmControl.compute_pick_plan()` 会计算完整抓取路径：

1. 抓取前预抬升 `pre_grasp_lift_mm`
2. 抓取点
3. 抓取后抬升 `post_grasp_lift_mm`

配置位置：

```json
"pre_grasp_lift_mm": 40.0,
"post_grasp_lift_mm": 100.0
```

以前抬升时保持同一个 `x/y`，如果目标已经比较远，直接竖直抬高可能导致 IK 超出可达范围，例如：

```text
Target wrist point is unreachable ... distance=307.0, reach=[5.0, 295.0]
```

现在新增了 `solve_lift_target()`：先尝试原地抬升；如果不可达，就沿当前 yaw 方向往回缩，每 5mm 重新求一次 IK。这样允许“上抬 + 回缩”，只要抓取可达，抬升阶段就会尽量找一个安全可达姿态。

运行时如果发生回缩，会看到类似：

```text
[IK] post_lift retracted r 415.1->390.1mm at z=265.0mm
```

这表示抓取后抬升高度保持目标值，但水平距离从 415.1mm 回缩到 390.1mm。

## 放置姿态

`place_block()` 现在不再只是把底座舵机回中后开爪，而是先计算一个正前方放置姿态：

```text
ArmControl.compute_place_pose()
```

放置目标：

- `yaw = 0`，机械臂正向前方
- 高度为桌面上方约 10cm
- 在该高度下尽量伸长
- 到位后再打开夹爪

配置位置：

```json
"place_height_above_table_mm": 100.0
```

桌面高度来自：

```json
"table_z_base_mm": 120.0
```

所以默认放置高度为：

```text
z = 120.0 + 100.0 = 220.0mm
```

如果现场发现放置太高或太低，优先调整 `Arm/params.json` 里的：

```json
"place_height_above_table_mm": 100.0
```

例如改为 `80.0` 表示桌面上方 8cm。

## 验证建议

修改舵机配置后：

```bash
python Arm/scan_servo_ids.py --ids 1-6 --baudrate 500000
```

运行任务前可以监控舵机：

```bash
python Arm/monitor_servos.py --ids 1 2 3 4 5 6 --single-line
```

如果某个 ID 仍然报错，先确认：

- 舵机是否供电
- 三线是否插牢、线序是否正确
- ID 是否正确
- 波特率是否为 `500000`
- 是否存在重复 ID
- 是否是同协议的 ST3215/SCServo 舵机
