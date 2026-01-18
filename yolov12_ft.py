# 使用本项目优化过的 YOLOv12 实现（非官方 ultralytics）
# 确保已运行: pip install -e .
from ultralytics import YOLO


if __name__ == '__main__':
    # --- 训练 YOLOv12s（与 YOLOv11s 对比）---
    model = YOLO("yolov12s.pt")  # 使用预训练权重
    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        batch=32,
        project="yolov12s_ft"
    )
