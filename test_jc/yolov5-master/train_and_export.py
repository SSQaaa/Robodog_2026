from ultralytics import YOLO
import yaml
import os

# ==============================
# 1. 配置区 (根据你的实际情况修改)
# ==============================

# 数据集路径 (你提供的路径)
TRAIN_IMAGES = "/home/ysc/Desktop/Robodog_2026/test_jc/new_doglabels.v6i.yolov5pytorch/train/images/"
VAL_IMAGES = "/home/ysc/Desktop/Robodog_2026/test_jc/new_doglabels.v6i.yolov5pytorch/valid/images/"
# 注意：YOLOv5 训练只需要 images 文件夹，labels 会自动在同级目录下寻找

# 类别信息 (你提供的 nc 和 names)
NC = 18
NAMES = ['1', '2', '3', '4', '5', '6', 'red_barrel', 'yellow_barrel', 'blue_barrel', 
         'orange_barrel', 'red_ball', 'yellow_ball', 'blue_ball', 'orange_ball', 
         'Dashboard', 'ssi', 'yellow_cylinder', 'red_cylinder']

# 项目设置
PROJECT_NAME = "runs/doglabels_training"
MODEL_TO_USE = "yolov5s.yaml" # 推荐用 s 或 m，n 模型太小可能精度不够，x 模型太大转 engine 麻烦

# ==============================
# 2. 代码逻辑 (无需修改)
# ==============================

def create_temp_data_yaml():
    """临时生成一个 data.yaml 文件"""
    data = {
        'train': TRAIN_IMAGES,
        'val': VAL_IMAGES,
        'nc': NC,
        'names': NAMES
    }
    path = "temp_dog_data.yaml"
    with open(path, 'w') as f:
        yaml.dump(data, f)
    return path

if __name__ == "__main__":
    print("?? 开始构建流程...")

    # --- 步骤 1: 准备数据配置 ---
    data_yaml_path = create_temp_data_yaml()
    print(f"?? 数据配置已生成: {data_yaml_path}")

    # --- 步骤 2: 初始化模型 ---
    # 注意：这里我们加载的是 YAML 文件（定义网络结构），而不是 PT 文件
    # 这样会随机初始化权重，适合从头训练你的数据
    model = YOLO(MODEL_TO_USE)

    # --- 步骤 3: 训练 (或者只做 1 个 epoch 验证数据通路) ---
    print("\n????♂? 开始训练...")
    # 如果你想真正训练模型，把 epochs 改为 100+
    # 如果你只是想测试数据集是否能跑通并转 ONNX，保留 1-3 个 epoch 即可
    results = model.train(
        data=data_yaml_path,
        epochs=100,       # 快速测试设为 1；正式训练建议 100+
        imgsz=640,      # 保持和你数据集导出时一致
        batch=16,       # 根据你的显存调整，10系/20系显卡建议 8-16
        name='exp',     # 实验名称
        project=PROJECT_NAME,
        exist_ok=True
    )

    # --- 步骤 4: 导出 ONNX (核心步骤) ---
    print("\n?? 正在导出 ONNX 模型...")
    
    # 获取刚刚训练保存的权重路径 (best.pt 或 last.pt)
    # YOLOv5 默认保存在 project/name/weights/best.pt
    weights_path = os.path.join(PROJECT_NAME, "exp", "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        print("?? 未找到训练权重，尝试使用 last.pt")
        weights_path = os.path.join(PROJECT_NAME, "exp", "weights", "last.pt")

    # 执行导出
    # simplify=True 是关键，它能清理模型结构，让 TensorRT 更容易解析
    success = model.export(
        format="onnx", 
        weights=weights_path, 
        imgsz=640, 
        simplify=True,  # 必须开启
        opset=11        # TensorRT 兼容性最好
    )

    if success:
        print(f"\n? 成功！ONNX 文件已生成。")
        print(f"?? 接下来，请使用 TensorRT 的 trtexec 工具转换 Engine：")
        print(f"   trtexec --onnx={weights_path.replace('.pt', '.onnx')} --saveEngine=model.engine --fp16")
    else:
        print("? 导出失败，请检查报错信息。")