#!/bin/bash

cd /home/ipr4090/2024_hzf/yolov12/yolov12
source .venv/bin/activate

echo "========== Validating yolov12n =========="
yolo val model=runs/VOC/yolov12n_voc_Downsample_DWT_all_4C_BNSiLU_b2h2+LL_with/weights/best.pt \
    data=datasets/VOC_handled/VOC_handled/VOC.yaml \
    batch=1 \
    imgsz=640 \
    device=0 \
    2>&1 | tee val_dwt_voc_n.txt

echo "========== Validating yolov12s =========="
yolo val model=runs/VOC/yolov12s_voc_Downsample_DWT_all_4C_BNSiLU_b2h2+LL_with/weights/best.pt \
    data=datasets/VOC_handled/VOC_handled/VOC.yaml \
    batch=1 \
    imgsz=640 \
    device=0 \
    2>&1 | tee val_dwt_voc_s.txt

echo "========== Validating yolov12m =========="
yolo val model=runs/VOC/yolov12m_voc_Downsample_DWT_all_4C_BNSiLU_b2h2+LL_with/weights/best.pt \
    data=datasets/VOC_handled/VOC_handled/VOC.yaml \
    batch=1 \
    imgsz=640 \
    device=0 \
    2>&1 | tee val_dwt_voc_m.txt

echo "========== All validations complete =========="
