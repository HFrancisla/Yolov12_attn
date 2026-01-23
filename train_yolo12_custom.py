"""
YOLOv12 自定义注意力 - 简化训练脚本
快速开始训练，支持切换不同注意力机制和模型大小
支持命令行参数: python train_yolo12_custom.py --attention MDTA --size s --epochs 200
"""
from ultralytics import YOLO
import argparse


# ============ 配置区域 ============
DEFAULT_MODEL_SIZE = "s"  # 可选: n, s, m, l, x
DEFAULT_ATTENTION_TYPE = "MDTA"  # 可选: MDTA, HTA, WTA, IRS, ICS
DEFAULT_EPOCHS = 200  # 推荐200轮，配合patience=50自动早停
DEFAULT_BATCH_SIZE = 32  # m模型建议16-32, l/x模型建议8-16
DEFAULT_DEVICE = 0  # 0表示第一个GPU，'cpu'表示使用CPU
DEFAULT_IMAGE_SIZE = 640
# =================================

# 配置文件（通用配置，支持所有模型大小和注意力类型）
CONFIG_FILE = "yolov12_custom.yaml"


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv12 自定义注意力训练脚本')
    parser.add_argument('--attention', type=str, default=None, 
                        help=f'注意力类型 (MDTA, HTA, WTA, IRS, ICS)，默认: {DEFAULT_ATTENTION_TYPE}')
    parser.add_argument('--size', type=str, default=None,
                        help=f'模型大小 (n, s, m, l, x)，默认: {DEFAULT_MODEL_SIZE}')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'训练轮数，默认: {DEFAULT_EPOCHS}')
    parser.add_argument('--batch', type=int, default=None,
                        help=f'批次大小，默认: {DEFAULT_BATCH_SIZE}')
    parser.add_argument('--device', default=None,
                        help=f'设备 (0, 1, cpu等)，默认: {DEFAULT_DEVICE}')
    parser.add_argument('--imgsz', type=int, default=None,
                        help=f'图像大小，默认: {DEFAULT_IMAGE_SIZE}')
    
    args = parser.parse_args()
    
    # 优先使用命令行参数，未指定则使用默认配置
    MODEL_SIZE = args.size if args.size is not None else DEFAULT_MODEL_SIZE
    ATTENTION_TYPE = args.attention if args.attention is not None else DEFAULT_ATTENTION_TYPE
    EPOCHS = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    BATCH_SIZE = args.batch if args.batch is not None else DEFAULT_BATCH_SIZE
    DEVICE = args.device if args.device is not None else DEFAULT_DEVICE
    IMAGE_SIZE = args.imgsz if args.imgsz is not None else DEFAULT_IMAGE_SIZE
    print(f"开始训练 YOLOv12{MODEL_SIZE} with {ATTENTION_TYPE} Attention")
    print(f"配置文件: {CONFIG_FILE}")
    
    # 读取并修改 YAML 配置
    import yaml
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新 scale 参数
    if 'scales' in config:
        config['scale'] = MODEL_SIZE
        print(f"使用模型大小: {MODEL_SIZE}")
    
    # 动态替换 backbone 中的注意力类型
    for i, layer in enumerate(config['backbone']):
        if len(layer) >= 4 and layer[2] == 'CustomA2C2f':
            # CustomA2C2f 的 YAML 参数格式: [c2, attn_type, residual, mlp_ratio]
            # (n 会被 parse_model 自动插入到索引2的位置)
            if len(layer[3]) >= 2:
                print(f"Layer {i} before: {layer[3]}")
                # 确保 attn_type 是字符串
                layer[3][1] = str(ATTENTION_TYPE)  # 替换注意力类型 (索引1)
                print(f"Layer {i} after: {layer[3]}")
    
    print(f"使用注意力机制: {ATTENTION_TYPE}")
    
    # 保存修改后的配置到临时文件
    temp_config = f"yolov12{MODEL_SIZE}_{ATTENTION_TYPE.lower()}_temp.yaml"
    with open(temp_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    # 创建模型
    model = YOLO(temp_config)
    
    # 开始训练（符合项目规范的参数）
    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        batch=BATCH_SIZE,
        workers=4,  # 减少 worker 数量，降低 pin_memory 内存占用
        # 优化器配置
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        # 数据增强
        scale=0.5,
        mosaic=1.0,
        mixup=0.15 if MODEL_SIZE in ['m', 'l', 'x'] else 0.0,  # 大模型使用mixup
        # 训练设置
        patience=50,
        project=f"runs/yolov12{MODEL_SIZE}_{ATTENTION_TYPE.lower()}",
        name="train"
    )
    
    print(f"\n训练完成！模型保存在: runs/yolov12{MODEL_SIZE}_{ATTENTION_TYPE.lower()}/train/weights/")
