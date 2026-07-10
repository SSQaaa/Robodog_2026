# AAA Robodog 2026

本项目是机器狗竞赛任务的整套控制程序，包含机器狗运动、路径规划、仪表盘识别、任务衔接、机械臂视觉标定以及物块抓取与放置。

主程序按以下顺序执行：

```text
任务 1 路径行走
    ↓
任务 2 仪表盘识别
    ↓
任务 2→3 场地衔接与定位
    ↓
机械臂复位
    ↓
任务 3 根据仪表状态抓取物块并放入对应箱体
```

> 本项目会真实控制机器狗和机械臂。首次部署时请分别完成通信、相机、模型、舵机和标定测试，不要直接运行 `main.py`。

## 1. 仓库结构

```text
.
├── main.py                       # 全流程入口：Task1 → Task2 → Task2_3 → Task3
├── project_config.py             # 全局配置：网络、模型、运动速度、任务参数
├── task1.py                      # 根据规划路径和世界坐标完成任务 1
├── task2.py                      # 仪表盘、字母和指针状态识别
├── task2_3.py                    # 任务 2 到任务 3 的定位与运动衔接
├── task3.py                      # 抓取、寻找目标箱、放置和返回流程
├── arm_control.py                # Task3 使用的机械臂控制封装
├── motion_test.py                # 机器狗基础运动测试
├── Arm/
│   ├── params.json               # 机械臂、相机、物块的实际运行参数
│   ├── config.py                 # params.json 的读取和保存
│   ├── calibration.py            # AprilTag 相机—机械臂外参标定
│   ├── vision_grasp.py           # 独立绿色物块检测/抓取测试
│   ├── kinematics.py             # 逆运动学、舵机目标值和安全检查
│   ├── camera_safety.py          # 机械臂与相机的碰撞安全检查
│   ├── servo_driver.py           # ST3215 舵机总线控制
│   ├── reset_arm.py              # 独立机械臂复位工具
│   ├── disable_servos.py         # 舵机扭矩启用/关闭工具
│   └── scservo_sdk/              # 舵机通信 SDK
└── tools/
    ├── motion.py                 # 机器狗 UDP 指令封装
    ├── vision.py                 # TensorRT YOLO、Orbbec 相机、仪表分析
    ├── world_pose.py             # ROS 世界坐标与 IMU 航向读取
    ├── task2_dashboard.py        # 仪表盘稳定识别辅助流程
    ├── task1_path_planner.py     # Task1 交互式路径规划工具
    ├── task1_path_plan.json      # Task1 实际使用的路径点
    └── task1_path_plan.svg       # 路径可视化
```

## 2. 运行环境与硬件

代码主要面向机器人端 Linux 环境，默认配置包括：

- 机器狗 UDP 地址：`192.168.1.120:43893`
- 机械臂串口：`/dev/ttyUSB0`
- 舵机波特率：`500000`
- Orbbec RGB-D 深度相机
- ST3215 系列总线舵机
- NVIDIA/TensorRT 推理环境
- ROS 世界坐标和 IMU 读取脚本

常规 Python 依赖至少包括：

```bash
pip install numpy opencv-python apriltag matplotlib
```

以下组件通常由机器人平台或相机/推理环境单独提供，不能只靠 `pip` 保证安装成功：

- `orbbec_native`：Orbbec 相机 Python 模块
- `yolov5_trt_cpp`：TensorRT YOLO Python 模块
- TensorRT、CUDA 及其系统动态库
- ROS 和项目使用的定位节点

视觉推理还要求项目根目录存在：

```text
libs/bigdog_0427.engine
libs/libmyplugins.so
```

当前代码通过 `project_config.py` 固定查找这两个文件。如果机器人端文件名或目录不同，需要同步修改 `ENGINE_PATH` 和 `PLUGIN_PATH`。

Task1 还会通过 Python 2 调用：

```text
task2/tasks/read_ros_world_position.py
task2/tasks/read_ros_imu_yaw.py
```

默认解释器为 `/usr/bin/python2`。部署前必须确认这两个脚本存在，或修改 `TASK1_WORLD_POSE_PYTHON` 及相关路径。当前仓库快照中没有看到这些运行资源，因此仅复制当前目录并不能直接完成整套实机任务。

## 3. 两套配置分别负责什么

### `project_config.py`

负责整机任务级配置：

- YOLO engine 和插件路径
- 类别 ID 与名称
- 仪表盘默认状态
- 机器狗 IP、端口与运动速度
- Task1 定位和路径参数
- Task2/Task3 使用的整机运动参数

文件中保留了一组 `ARM_*` 常量，但当前 Task3 的 `ArmControl` 实际从 `Arm/params.json` 读取机械臂参数。调整机械臂实机参数时，应以 `Arm/params.json` 为准。

### `Arm/params.json`

这是机械臂当前实际使用的默认参数文件，包含：

- `camera`：AprilTag 信息、`T_base_tag`、标定得到的 `T_base_camera`
- `block`：物块尺寸、桌面高度、抓取补偿、绿色 HSV 范围
- `arm`：串口、波特率、舵机 ID、连杆长度、零位、方向、限位
- `arm.gripper`：夹爪开合位置、电流保护和闭合策略
- `arm.reset_pose`：机械臂复位姿态
- `arm.camera_safety`：相机碰撞安全区域

`arm_control.py` 现在只加载项目根目录下的 `Arm/`，不会再查找 `task3/Arm/`。

## 4. 首次部署建议顺序

### 4.1 检查机器狗通信

确认电脑和机器狗网络互通，并检查：

```python
DOG_IP = "192.168.1.120"
DOG_PORT = 43893
```

运动测试会真实移动机器狗。清空周围区域并准备急停后运行：

```bash
python motion_test.py
```

注意：该脚本当前连续执行多次前进，不是单次轻微点动。建议先检查并减小文件中的速度与时间。

### 4.2 检查机械臂串口

Linux 下确认串口：

```bash
ls -l /dev/ttyUSB*
```

如有权限问题，可临时测试：

```bash
sudo chmod 666 /dev/ttyUSB0
```

长期部署建议通过用户组或 udev 规则授予权限，而不是每次执行 `chmod`。

### 4.3 核对舵机 ID、限位和复位姿态

机械臂会使用 `Arm/params.json` 中的 ID 1～6。第一次运动前，应核对：

- 每个关节对应的舵机 ID
- `zero` 与 `direction`
- 每个舵机的 `min`/`max`
- `reset_pose` 是否在安全范围内
- 机械臂移动路径是否会碰撞机身、相机或地面

执行复位：

```bash
python Arm/reset_arm.py
```

复位时同时采用夹爪打开位置：

```bash
python Arm/reset_arm.py --open-gripper
```

关闭所有已配置舵机的扭矩：

```bash
python Arm/disable_servos.py
```

重新启用扭矩：

```bash
python Arm/disable_servos.py --enable
```

## 5. 相机—机械臂标定

### 5.1 标定原理

`Arm/calibration.py` 使用固定的 AprilTag 计算相机坐标系到机械臂基座坐标系的变换：

```text
T_base_camera = T_base_tag × inverse(T_camera_tag)
```

标定结果保存到：

```text
Arm/params.json → camera.T_base_camera
```

### 5.2 标定前准备

1. 使用 `tag36h11` 标签，默认 ID 为 `0`。
2. 确认标签实际边长与 `camera.tag_size_mm` 一致，当前默认值为 `26.0 mm`。
3. 将标签固定在机械臂基座坐标系中的已知位置和朝向。
4. 将准确的标签位姿写入 `camera.T_base_tag`。
5. 固定相机和机械臂。标定后相机位置或角度不能再改变。
6. 保证标签清晰、无遮挡、光照稳定。

`T_base_tag` 如果不准确，即使画面中的标签识别很稳定，最终抓取坐标也会整体偏移。

### 5.3 先预览但不保存

```bash
python Arm/calibration.py --samples 60 --show
```

不加 `--save` 时只计算和打印结果，不会修改参数文件。

### 5.4 正式标定并保存

```bash
python Arm/calibration.py --samples 60 --save --show
```

无显示器或 SSH 环境下去掉 `--show`：

```bash
python Arm/calibration.py --samples 60 --save
```

主要参数：

- `--samples`：总采样数，默认 60
- `--min-samples`：最少有效样本数，默认 20
- `--sample-delay`：采样间隔，默认 0.05 秒
- `--max-sample-error-mm`：平移离群样本阈值，默认 20 mm
- `--save`：把结果写入 `params.json`
- `--show`：显示 AprilTag 检测窗口
- `--config PATH`：使用其他参数文件

保存后建议记录输出中的有效样本数、剔除后样本数和平移标准差。标准差较大时，应先改善标签固定、光照和相机稳定性，再重新标定。

## 6. 独立抓取测试

`Arm/vision_grasp.py` 是机械臂的独立测试入口。它通过 HSV 检测绿色物块、读取深度、转换到机械臂基座坐标、求逆运动学，并可选择执行抓取。

它不是 Task3 的完整入口，并且默认只识别绿色物块，不会执行机器狗寻找物块或箱体的流程。

### 6.1 只检测和计算，不移动机械臂

推荐首次运行：

```bash
python Arm/vision_grasp.py --dry-run --show
```

也可以直接运行：

```bash
python Arm/vision_grasp.py
```

未添加 `--execute` 时，程序默认不会发送抓取动作，只打印识别结果、目标角度和舵机位置。它可能尝试读取舵机当前状态；读取失败会打印提示，但不会因此执行动作。

### 6.2 真正执行抓取

确认标定结果、坐标、逆运动学结果和运动空间安全后运行：

```bash
python Arm/vision_grasp.py --execute
```

需要同时查看检测画面时：

```bash
python Arm/vision_grasp.py --execute --show
```

真实执行顺序为：

```text
打开夹爪 → 移动到预抓取点 → 下探到抓取点
→ 电流保护闭合夹爪 → 抬升 → 打印最终舵机状态
```

不要同时使用 `--dry-run` 和 `--execute`，程序会直接报错。

脚本默认会重新获取并检测最多 60 帧，以等待彩色流和深度流同时就绪。可按相机启动速度调整：

```bash
python Arm/vision_grasp.py --dry-run --max-attempts 100 --retry-delay 0.05
```

- `--max-attempts`：最大取帧/深度查询次数，默认 60
- `--retry-delay`：每次重试间隔秒数，默认 0.05

### 6.3 常见抓取调整项

抓取发生固定偏移时，优先按以下顺序排查：

1. 相机是否在标定后发生移动。
2. `T_base_tag` 和标签尺寸是否正确。
3. 深度值是否稳定。
4. `table_z_base_mm` 和物块 `size_mm` 是否符合实际。
5. 最后再微调 `grasp_offset_base_mm` 和 `grasp_z_offset_mm`。

绿色检测不稳定时调整：

```text
block.hsv_lower
block.hsv_upper
block.min_area_px
block.morph_kernel_px
```

## 7. 逆运动学测试

测试目标点 `(x, y, z)`，单位为毫米：

```bash
python Arm/kinematics.py --x 300 --y 0 --z 150 --dry-run
```

该脚本会计算舵机目标值、检查相机安全区域，并显示机械臂姿态。虽然参数名包含 `--dry-run`，脚本仍会尝试读取舵机当前位置；它不会发送关节运动命令。

也可以直接查看指定关节角姿态：

```bash
python Arm/kinematics.py --theta1 30 --theta2 40 --theta3 20
```

## 8. 各任务说明

### Task1：路径行走

入口函数：

```python
task1.run(dog)
```

Task1 读取 `tools/task1_path_plan.json` 中的毫米路径点，再通过 ROS 世界坐标和 IMU 航向将其映射到机器人起始坐标系，分段执行前后/横向运动，并在每段后进行位置修正。

重新生成或编辑路径规划：

```bash
python tools/task1_path_planner.py
```

具体选项可查看：

```bash
python tools/task1_path_planner.py --help
```

### Task2：仪表盘识别

入口函数：

```python
task2.run(dog)
```

Task2 使用 TensorRT YOLO 检测字母、仪表盘和指针区域，并对仪表状态进行稳定判断。返回记录会由 `main.py` 整理成 A/B/C/D 状态，传给 Task3。

Task2 没有独立的命令行 `main`，通常由总流程调用。

### Task2_3：任务衔接

入口函数：

```python
task2_3.run(dog, start_yaw_deg=start_yaw_deg)
```

该阶段继续使用仪表/字母视觉检测，并结合起始航向完成 Task2 到 Task3 区域的对准、距离调整和衔接运动。

### Task3：抓取与投放

可独立启动：

```bash
python task3.py
```

但这会使用 `project_config.py` 中的默认仪表状态，并真实启动相机、机器狗和机械臂。Task3 的流程包括：

1. 根据 A/B/C/D 状态判断需要处理的异常目标。
2. 识别相应颜色物块并调整机器狗位置。
3. 使用 `ArmControl` 计算和执行抓取。
4. 机器狗转向并寻找对应字母箱体。
5. 接近箱体并调用机械臂放置。
6. 如有下一个异常目标，则返回抓取区继续执行。

其中独立 `Arm/vision_grasp.py` 使用 HSV 识别绿色物块，而完整 Task3 使用 TensorRT YOLO 的 `Green`/`Red` 类别，两者不要混为同一个视觉入口。

## 9. 运行完整流程

完成所有单项测试后，在项目根目录执行：

```bash
python main.py
```

`main.py` 会：

1. 创建 `DogControl`，站立并停止连续运动。
2. 读取初始 IMU 航向。
3. 执行 Task1。
4. 执行 Task2；若失败则使用默认仪表状态。
5. 执行 Task2_3 衔接。
6. 机械臂复位。
7. 创建并执行 Task3。
8. 异常或退出时停止机器狗并关闭相机、舵机和 UDP 连接。

## 10. 安全注意事项

- 首次调试必须使用低速、短时间和宽敞环境。
- 每次真实抓取前先运行 `vision_grasp.py --dry-run`。
- `--show` 仅用于有图形界面的环境；无显示器时不要使用。
- 标定和抓取期间不要移动相机、标签或机械臂安装座。
- 不要仅依赖软件舵机限位；首次动作应随时准备断电。
- 修改舵机 `zero`、`direction`、限位或连杆长度后必须重新验证逆运动学。
- `camera_safety` 只检查配置中建模的区域，不能覆盖所有真实碰撞情况。
- 运行 `main.py`、`task3.py`、`reset_arm.py` 或带 `--execute` 的抓取命令前，确认串口对应的是正确设备。

## 11. 常见问题

### `ModuleNotFoundError: orbbec_native`

Orbbec 相机运行库没有安装或不在 `PYTHONPATH`。需要按当前机器人平台的 Orbbec SDK 部署方式安装对应模块。

### 找不到 `libmyplugins.so` 或 `bigdog_0427.engine`

确认 `libs/` 目录和文件存在，并与机器人端 TensorRT/CUDA 版本匹配。TensorRT engine 通常不能在不同平台或不兼容版本之间直接通用。

### 找不到世界坐标读取脚本

确认以下文件部署完整：

```text
task2/tasks/read_ros_world_position.py
task2/tasks/read_ros_imu_yaw.py
```

同时确认 ROS 环境已启动，且 `/usr/bin/python2` 能运行这些脚本。

### 标定成功但抓取有固定偏差

重点检查 `T_base_tag`、标签边长、桌面高度、物块高度以及相机是否在标定后移动。固定偏差最后才用 `grasp_offset_base_mm` 补偿。

### `camera.T_base_camera missing`

执行：

```bash
python Arm/calibration.py --samples 60 --save
```

并确认结果确实写入当前项目的 `Arm/params.json`。

### 串口打不开

确认设备名、USB 连接、用户权限以及是否有其他进程占用串口。默认设备配置在 `Arm/params.json` 的 `arm.devicename`。

### 目标不可达或触发安全检查

检查目标坐标、连杆长度、舵机角度限位和相机安全区域。不要通过盲目扩大限位绕过真实机械约束。

## 12. 推荐的实机验收清单

- [ ] 机器狗 IP 和 UDP 端口正确
- [ ] `motion_test.py` 低速测试通过
- [ ] ROS 世界坐标和 IMU 脚本可单独运行
- [ ] TensorRT engine、插件和 Python 模块可加载
- [ ] Orbbec 彩色帧、深度和内参读取正常
- [ ] 舵机 ID、方向、限位与复位姿态核对完成
- [ ] AprilTag 尺寸和 `T_base_tag` 实测无误
- [ ] 标定结果已用 `--save` 写入 `Arm/params.json`
- [ ] 独立抓取 dry-run 的目标坐标和舵机值合理
- [ ] 低速真实抓取测试通过
- [ ] Task1、Task2、Task2_3、Task3 分阶段测试通过
- [ ] 最后再执行 `python main.py`
