from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov12n-dwt.yaml")

    model.train(
        data="datasets/VOC_handled/VOC_handled/VOC.yaml",
        epochs=200,
        imgsz=640,
        batch=64,
        workers=8,
        device=0,
        name="yolov12n_voc_Downsample_DWT_all_4C_BNSiLU+LL_with",
        project="runs/VOC",
    )
