"""
YOLOv12 自定义注意力 - 简化训练脚本
快速开始训练，支持切换不同注意力机制和模型大小
"""
from ultralytics import YOLO


# ============ 配置区域 ============
MODEL_SIZE = "s"  # 可选: n, s, m, l, x
ATTENTION_TYPE = "MDTA"  # 可选: MDTA, HTA, WTA, IRS, ICS
EPOCHS = 200  # 推荐200轮，配合patience=50自动早停
BATCH_SIZE = 32  # m模型建议16-32, l/x模型建议8-16
DEVICE = 0  # 0表示第一个GPU，'cpu'表示使用CPU
IMAGE_SIZE = 640
# =================================

# 配置文件（通用配置，支持所有模型大小和注意力类型）
CONFIG_FILE = "yolov12_custom.yaml"


if __name__ == '__main__':
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
    for layer in config['backbone']:
        if len(layer) >= 4 and layer[2] == 'CustomA2C2f':
            # CustomA2C2f 的参数格式: [channels, shortcut, attention_type, ...]
            if len(layer[3]) >= 3:
                layer[3][2] = ATTENTION_TYPE  # 替换注意力类型
    
    print(f"使用注意力机制: {ATTENTION_TYPE}")
    
    # 创建模型
    model = YOLO(config)  # 直接传入字典
    
    # 开始训练（符合项目规范的参数）
    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        batch=BATCH_SIZE,
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
