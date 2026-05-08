from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov12m-dwt.yaml")

    model.train(
        data="datasets/VOC_handled/VOC_handled/VOC.yaml",
        epochs=200,
        imgsz=640,
        batch=32,
        workers=8,
        device=0,
        name="yolov12m_voc_Downsample_DWT_all_4C_to_2C_without",
        project="runs/VOC",
    )
