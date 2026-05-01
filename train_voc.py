from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov12m.yaml")

    model.train(
        data="datasets/VOC_handled/VOC.yaml",
        epochs=200,
        imgsz=640,
        batch=32,
        device=0,
        name="yolov12m_voc_baseline",
    )
