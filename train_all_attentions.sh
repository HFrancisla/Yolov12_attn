#!/bin/bash
# YOLOv12 多注意力机制训练脚本
# 依次训练 MDTA, WTA, HTA 三种注意力机制
# 每个训练的控制台输出保存到对应的 txt 文件

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}YOLOv12 多注意力机制训练脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 定义注意力类型列表
attention_types=("MDTA" "WTA" "HTA")

# 创建日志目录
log_dir="training_logs"
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    echo -e "${GREEN}创建日志目录: $log_dir${NC}"
fi

# 遍历每种注意力类型进行训练
for attention in "${attention_types[@]}"; do
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}开始训练: $attention${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    
    # 定义日志文件路径
    log_file="$log_dir/train_${attention}.txt"
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 写入训练开始信息到日志文件
    echo "训练开始时间: $timestamp" > "$log_file"
    echo "注意力类型: $attention" >> "$log_file"
    echo "================================================================================" >> "$log_file"
    echo "" >> "$log_file"
    
    echo -e "${GREEN}使用注意力类型: $attention${NC}"
    echo -e "${GREEN}日志文件: $log_file${NC}"
    echo ""
    
    # 执行训练并保存输出（使用命令行参数）
    if python train_yolo12_custom.py --attention "$attention" 2>&1 | tee -a "$log_file"; then
        end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "" >> "$log_file"
        echo "================================================================================" >> "$log_file"
        echo "训练结束时间: $end_timestamp" >> "$log_file"
        
        echo ""
        echo -e "${GREEN}✓ $attention 训练完成${NC}"
        echo -e "${GREEN}日志已保存到: $log_file${NC}"
    else
        echo ""
        echo -e "${RED}✗ $attention 训练失败${NC}"
        echo "训练失败: 退出码 $?" >> "$log_file"
    fi
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}所有训练任务完成！${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${GREEN}训练日志保存在: $log_dir 目录${NC}"
echo "- train_MDTA.txt"
echo "- train_WTA.txt"
echo "- train_HTA.txt"
