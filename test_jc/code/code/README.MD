# 机器狗国赛线下挑战赛

## 项目概述
本项目是一个集成了机器狗运动控制、计算机视觉检测和语音反馈。系统通过TensorRT+YOLOv5目标检测模型识别环境中的各种元素（仪表盘、锥桶、数字和小球），并根据检测结果控制机器狗完成巡检、救援和抓取三大任务。

## 主要功能
- **巡检任务**：检测仪表盘状态、识别纸箱数字和锥桶颜色
- **救援任务**：追踪并踢动异常区域对应的彩色球
- **抓取任务**：二维码定位并使用机械臂完成抓取
- **语音反馈**：实时播报检测结果


## 文件结构
```plaintext
├── detect.py                  # 元素检测与状态判断核心逻辑
├── detect_trt.py              # TensorRT加速的YOLOv5检测实现
├── dog_control.py             # 机器狗控制主程序（运动控制+任务流程）
├── sound_output.py            # 语音反馈模块
├── udp.py                     # UDP通信模块
├── yolov5trt.py               # YOLOv5 TensorRT引擎封装
├── yolov5/                    # 适用于python3.6.9的YOLOv5源码目录 
    ├── best.pt                # 模型文件
    └── detect.py              # 模型测试文件
└── sound/                     # 语音资源目录
    ├── red_barrel.mp3
    ├── orange_barrel.mp3
    └── ... (其他语音文件)
```

## 依赖环境
- Python 3.6
- Torch
- TorchVision
- OpenCV
- PyCUDA
- TensorRT
- NumPy
- playsound
- pyzbar

## 硬件要求
- NVIDIA GPU (支持CUDA)
- 支持UDP控制的机器狗
- USB摄像头

## 运行指南

### 安装依赖
```bash
pip install opencv-python torch torchvision pycuda numpy playsound pyzbar
```

### 运行主程序
```bash
python dog_control.py
```

### 控制流程
1. 机器狗启动并站立
2. 执行巡检任务（检测纸箱元素）
3. 执行救援任务（追踪并踢球）
4. 执行抓取任务（识别二维码并实现抓取）

## 关键模块说明

### 1. 运动控制 (`dog_control.py`)
- `move()`: 基础运动控制
- `EXmove()`/`NEXmove()`: 特殊步态运动
- `shake_head()`/`nod_head()`: 头部动作
- `revolve_180()`/`revolve_90()`: 旋转控制

### 2. 视觉检测 (`detect.py`, `detect_trt.py`)
- 仪表盘状态检测 (正常/高/低)
- 数字识别 (1-6)
- 锥桶颜色识别 (红/黄/蓝/橙)
- 二维码识别

### 3. 任务流程 (`dog_control.py`)
- `Process_One()`: 巡检任务流程
- `Process_Two()`: 救援任务流程
- `Process_Three()`: 抓取任务流程

### 4. 语音反馈 (`sound_output.py`)
- 根据检测结果播放对应语音
- 支持锥桶颜色、数字和仪表盘状态的语音反馈

### 5. 模型测试 (`./yolov5/detect.py`)
- 检测YOLOv5模型best.pt效果

## 配置说明
1. 在`udp.py`中配置机器狗的IP和端口
2. 在`dog_control.py`中配置摄像头索引
3. TensorRT需在对应主机环境下转换模型才可使用，该工程提供了YOLOv5模型测试代码`./yolov5/detect.py`
4. 确保`sound/`目录中有所有需要的语音文件
