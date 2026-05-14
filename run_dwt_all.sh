#!/bin/bash
PYTHON="/home/ipr4090/2024_hzf/yolov12/yolov12/.venv/bin/python"
DIR="/home/ipr4090/2024_hzf/yolov12/yolov12"

$PYTHON "$DIR/train_dwt_voc_m_32.py"
$PYTHON "$DIR/train_dwt_voc_n_64.py"
$PYTHON "$DIR/train_dwt_voc_s_64.py"
