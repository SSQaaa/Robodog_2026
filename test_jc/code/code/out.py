import sys
import os

# 添加 yolo11.py 所在目录到路径
sys.path.insert(0, "/path/to/yolo11")

import numpy as np

# 直接导入
try:
    from yolo11 import YoLov5TRT
    print("导入成功")
except ImportError as e:
    print(f"导入失败: {e}")
    # 尝试查看 yolo11.py 中的类名
    import yolo11
    print(f"yolo11 中的属性: {dir(yolo11)}")
    sys.exit(1)

# 检查引擎文件
engine_path = "yolo11n.engine"
if not os.path.exists(engine_path):
    print(f"引擎文件不存在: {engine_path}")
    print(f"当前目录: {os.getcwd()}")
    print(f"文件列表: {os.listdir('.')}")
    sys.exit(1)

# 创建检测器
try:
    detector = YoLov5TRT(engine_path)
    print("检测器创建成功")
except Exception as e:
    print(f"创建检测器失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试推理
try:
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    output = detector.infer(dummy_input)
    print(f"推理成功")
    print(f"输出类型: {type(output)}")
    print(f"输出形状: {np.array(output).shape}")
    print(f"前10个值: {np.array(output).flatten()[:10]}")
except Exception as e:
    print(f"推理失败: {e}")
    import traceback
    traceback.print_exc()