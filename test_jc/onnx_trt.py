# -*- coding: utf-8 -*-
import tensorrt as trt
import os

def build_engine_trt7(onnx_path, engine_path="best.engine", fp16=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    # TRT7.1 用 EXPLICIT_BATCH 模式
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"解析错误: {parser.get_error(err)}")
            return None

    # 构建配置（TRT7.1 语法）
    builder.max_workspace_size = 1 << 30  # 1GB 显存
    if fp16:
        builder.fp16_mode = True
    builder.max_batch_size = 1

    # 构建 Engine
    engine = builder.build_cuda_engine(network)
    if not engine:
        print("构建 Engine 失败")
        return None

    # 保存
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"? 生成适配 TRT7.1 的 Engine: {engine_path}")
    return engine_path

if __name__ == "__main__":
    build_engine_trt7("best.onnx", "best.engine", fp16=True)