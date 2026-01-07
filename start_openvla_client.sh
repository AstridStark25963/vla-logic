#!/bin/bash

echo "========================================"
echo "  OpenVLA 控制客户端启动脚本"
echo "========================================"

# ===== 配置区域 =====
# 根据实际情况修改

# ROS Master 地址
export ROS_MASTER_URI=http://192.168.1.12:11311

# 本机 IP
export ROS_IP=192.168.1.12

# 相机 ID
LEFT_CAM_ID=4
RIGHT_CAM_ID=10

# ===== 环境设置 =====

# 加载 ROS 环境
if [ -f "/opt/ros/noetic/setup. bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo "✓ ROS Noetic 环境已加载"
else
    echo "✗ ROS 环境未找到"
    exit 1
fi

# 添加 ROS Python 路径（如果需要）
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH

# 切换到脚本目录
cd "$(dirname "$0")"

# ===== 运行环境检查 =====

echo ""
echo "运行环境检查..."
python3 check_environment.py $LEFT_CAM_ID $RIGHT_CAM_ID

if [ $? -ne 0 ]; then
    echo ""
    read -p "环境检查未完全通过，是否继续？ [y/N]:  " continue_anyway
    if [[ !  "$continue_anyway" =~ ^[Yy]$ ]]; then
        echo "已取消启动"
        exit 1
    fi
fi

# ===== 启动客户端 =====

echo ""
echo "启动 OpenVLA 控制客户端..."
echo "按 Ctrl+C 停止"
echo "========================================"
echo ""

python3 openvla_control_client.py

echo ""
echo "OpenVLA 控制客户端已停止"
