"""
训练原始 YOLOv12s 模型（从头训练）
使用 Ultralytics 官方的 YOLOv12 架构，不使用自定义注意力机制
"""
from ultralytics import YOLO


# ============ 配置区域 ============
MODEL_SIZE = "s"  # 模型大小
EPOCHS = 200  # 训练轮数
BATCH_SIZE = 32  # 批次大小
DEVICE = 0  # GPU设备
IMAGE_SIZE = 640  # 图像大小
# =================================


if __name__ == '__main__':
    print(f"开始从头训练原始 YOLOv12{MODEL_SIZE}")
    
    # 使用官方配置文件（YOLOv12 使用 scales 机制，通过 yolov12{size}.yaml 自动选择对应的 scale）
    config_file = f"yolov12{MODEL_SIZE}.yaml"
    
    # 创建模型（从头训练）
    model = YOLO(config_file)
    
    # 开始训练（符合项目规范的参数）
    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        batch=BATCH_SIZE,
        workers=2,
        # 优化器配置
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        # 数据增强
        scale=0.5,
        mosaic=1.0,
        mixup=0.0,  # s模型不使用mixup
        # 训练设置
        patience=50,
        project=f"runs/yolov12{MODEL_SIZE}_original",
        name="train"
    )
    
    print(f"\n训练完成！模型保存在: runs/yolov12{MODEL_SIZE}_original/train/weights/")
    print(f"最佳模型: runs/yolov12{MODEL_SIZE}_original/train/weights/best.pt")
    print(f"最后模型: runs/yolov12{MODEL_SIZE}_original/train/weights/last.pt")
