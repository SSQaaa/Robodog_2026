# Task3 搬运任务

本目录用于完成任务三：根据仪表盘状态抓取红色或绿色物块，并把物块放到对应的 A/B/C/D 箱子上。

主流程入口是 `task3.py`。代码按职责拆开：

- `task3.py`：任务编排，只负责把视觉、机械臂、机器狗运动串起来。
- `vision_control.py`：视觉识别，只负责 TensorRT YOLO + Orbbec 深度，输出目标类别、中心点、框和深度。
- `arm_control.py`：机械臂控制，只负责把视觉目标转换成机械臂目标点，并执行抓取/松爪。
- `dog_control.py`：机器狗运动控制，只负责 UDP 运动指令。
- `Arm/`：最新机械臂程序和参数，包括 IK、舵机驱动、标定、复位和单独抓取测试。
- `obj_det.py`：YOLO + 深度检测参考脚本，可单独验证模型和相机。

## 运行主任务

在机器人上进入本目录或项目根目录后运行：

```bash
python task3.py
```

或者从项目根目录运行：

```bash
python 2026Project/task3/task3.py
```

`task3.py` 不使用命令行参数。比赛现场需要改参数时，直接修改文件顶部常量：

```python
DASHBOARD_STATUS = {
    "A": "异常",
    "B": "正常",
    "C": "异常",
    "D": "正常",
}

ARM_DRY_RUN = False
INITIAL_BOX_FORWARD_SECONDS = 2.0
RETURN_FORWARD_SECONDS = 5.0
CENTER_TOLERANCE_PX = 60
BOX_TARGET_DEPTH_MM = 600
```

状态映射规则：

- `异常` -> 抓 `Red`
- `正常` -> 抓 `Green`

## 主流程

每个字母按 `A/B/C/D` 顺序执行：

1. 根据状态选择 `Red` 或 `Green`。
2. 视觉检测对应颜色物块，并取得深度。
3. 机械臂抓取物块。
4. 机器狗原地转 180 度。
5. 向箱子方向先走 `2s`。
6. YOLO 检测目标字母 `A/B/C/D`。
7. 左右平移，直到字母接近画面中心。
8. 根据字母深度向前靠近箱子。
9. 机械臂松爪放下物块。
10. 机器狗原地转 180 度。
11. 直走 `5s` 回到取物区，继续下一轮。

## 机械臂参数和工具

机械臂最新代码在 `Arm/` 目录中。主任务读取：

```text
Arm/params.json
```

其中最重要的是：

- `camera.T_base_camera`：相机到机械臂基座的标定矩阵。
- `block`：物块尺寸、桌面高度、抓取偏移等。
- `arm`：舵机 ID、连杆长度、舵机限位、抓手参数。

常用机械臂命令需要在 `Arm/` 目录下执行：

```bash
cd Arm
python reset_arm.py
python reset_arm.py --open-gripper
python monitor_servos.py --ids 3 4 5 --single-line
python disable_servos.py
python disable_servos.py --enable --ids 3 4 5
```

标定相机到机械臂基座：

```bash
cd Arm
python calibration.py --samples 60 --save
```

单独测试机械臂视觉抓取：

```bash
cd Arm
python vision_grasp.py --dry-run
python vision_grasp.py --execute
```

## 视觉检测

任务三主流程使用 `vision_control.py`，类别表来自当前 YOLO 模型：

```python
0: "A"
1: "B"
2: "C"
3: "D"
4: "Green"
6: "Red"
```

如果要单独验证 YOLO 和深度是否正常，可以运行：

```bash
python obj_det.py
```

确认能稳定看到：

- `Red`
- `Green`
- `A/B/C/D`
- 合理的 `depth_mm`

## 调试建议

推荐顺序：

1. 先跑 `obj_det.py`，确认 YOLO 类别和深度正常。
2. 再在 `Arm/` 下跑 `vision_grasp.py --dry-run`，确认机械臂抓取点和 IK 正常。
3. 再跑 `vision_grasp.py --execute`，确认机械臂能单独抓取。
4. 再把 `task3.py` 里的 `ARM_DRY_RUN = True`，只空跑机械臂，检查抓取目标和 IK 是否正常。注意机器狗运动控制没有 dry-run，会真实发送 UDP。
5. 最后设置 `ARM_DRY_RUN = False`，让机械臂和机器狗都实车运行。

## 注意事项

- `task3.py` 只编排流程，不在里面写底层机械臂或视觉细节。
- 修改机器狗速度时，优先改 `dog_control.py` 顶部常量。
- 修改任务流程时间和距离阈值时，改 `task3.py` 顶部常量。
- 修改机械臂几何、舵机、抓取高度和标定矩阵时，改 `Arm/params.json`。
- 当前返回取物区采用固定动作：转 180 度后直走 `5s`，不做视觉定位。
