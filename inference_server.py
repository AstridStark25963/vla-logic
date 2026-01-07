#!/usr/bin/env python3
"""
推理服务器 - 在主机A上运行
主机A IP: 192.168.1.10
ROS Master: 192.168.1.12 (主机B)

功能:
1. 订阅来自控制客户端的图像和机器人状态
2. 运行TinyVLA模型推理
3. 发布动作序列到控制客户端
"""

import os
import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import torch
import numpy as np
import pickle
import time
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_real_franka import llava_pythia_act_policy


def setup_logging():
    """配置日志系统：同时输出到终端和文件"""
    # 创建日志文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"inference_server_{timestamp}.log"

    # 配置日志格式
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 写入文件
            logging.StreamHandler(sys.stdout)  # 输出到终端
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")

    # 重定向stdout和stderr到日志文件（同时保留终端输出）
    class TeeOutput:
        """同时写入多个输出流"""
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    # 保存原始stdout和stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 打开日志文件用于写入print输出
    log_fileobj = open(log_file, 'a', encoding='utf-8')

    # 重定向stdout和stderr到文件和终端
    sys.stdout = TeeOutput(original_stdout, log_fileobj)
    sys.stderr = TeeOutput(original_stderr, log_fileobj)

    return logger


class InferenceServer:
    def __init__(self, policy_config, task_description="pick up the white bowl"):
        """
        初始化推理服务器

        Args:
            policy_config: 策略模型配置
            task_description: 任务描述
        """
        # 初始化ROS节点
        rospy.init_node('tinyvla_inference_server', anonymous=False)
        rospy.loginfo("正在初始化推理服务器...")

        # 配置参数
        self.policy_config = policy_config
        self.task_description = task_description

        # 加载模型
        rospy.loginfo("正在加载TinyVLA模型...")
        self.policy = llava_pythia_act_policy(self.policy_config)
        rospy.loginfo("模型加载完成")

        # 加载数据集统计信息
        stats_path = os.path.join(self.policy_config['model_path'], 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        rospy.loginfo("数据集统计信息加载完成")

        # 初始化CV Bridge
        self.bridge = CvBridge()

        # 数据缓存
        self.left_image = None
        self.right_image = None
        self.robot_state = None

        # 推理控制
        self.last_inference_time = 0
        self.inference_interval = 1.0  # 1Hz推理频率
        self.min_inference_interval = 0.5  # 最小推理间隔

        # 性能统计
        self.inference_count = 0
        self.total_inference_time = 0

        # ROS订阅者
        rospy.loginfo("正在设置ROS订阅者...")
        self.image_left_sub = rospy.Subscriber(
            '/camera/left/image_raw',
            Image,
            self.image_left_callback,
            queue_size=1,
            buff_size=2**24  # 增加缓冲区大小
        )
        self.image_right_sub = rospy.Subscriber(
            '/camera/right/image_raw',
            Image,
            self.image_right_callback,
            queue_size=1,
            buff_size=2**24
        )
        self.state_sub = rospy.Subscriber(
            '/robot/state',
            Float32MultiArray,
            self.state_callback,
            queue_size=1
        )

        # ROS发布者
        rospy.loginfo("正在设置ROS发布者...")
        self.action_pub = rospy.Publisher(
            '/inference/actions',
            Float32MultiArray,
            queue_size=1
        )

        # 性能监控定时器
        rospy.Timer(rospy.Duration(10.0), self.print_stats)

        rospy.loginfo("=" * 60)
        rospy.loginfo("推理服务器启动完成")
        rospy.loginfo("主机A IP: 192.168.1.10")
        rospy.loginfo("ROS Master: 192.168.1.12")
        rospy.loginfo("任务描述: %s", self.task_description)
        rospy.loginfo("推理频率: %.1f Hz", 1.0 / self.inference_interval)
        rospy.loginfo("=" * 60)

    def image_left_callback(self, msg):
        """接收左相机图像"""
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            rospy.logdebug("接收到左相机图像: %s", self.left_image.shape)
        except Exception as e:
            rospy.logerr("左相机图像转换错误: %s", e)

    def image_right_callback(self, msg):
        """接收右相机图像"""
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            rospy.logdebug("接收到右相机图像: %s", self.right_image.shape)
        except Exception as e:
            rospy.logerr("右相机图像转换错误: %s", e)

    def state_callback(self, msg):
        """
        接收机器人状态并触发推理

        Args:
            msg: Float32MultiArray消息,包含机器人关节状态
        """
        try:
            self.robot_state = np.array(msg.data)
            rospy.logdebug("接收到机器人状态: %s", self.robot_state)

            # 检查是否可以进行推理
            current_time = rospy.get_time()
            time_since_last = current_time - self.last_inference_time

            if time_since_last >= self.inference_interval:
                self.run_inference()
                self.last_inference_time = current_time
            else:
                rospy.logdebug("推理间隔不足,跳过 (%.2fs < %.2fs)",
                             time_since_last, self.inference_interval)
        except Exception as e:
            rospy.logerr("状态回调错误: %s", e)

    def run_inference(self):
        """运行模型推理"""
        # 检查数据完整性
        if self.left_image is None or self.right_image is None or self.robot_state is None:
            rospy.logwarn("等待数据... (左相机: %s, 右相机: %s, 状态: %s)",
                         self.left_image is not None,
                         self.right_image is not None,
                         self.robot_state is not None)
            return

        try:
            start_time = time.time()

            # 准备输入数据
            images = np.stack([self.left_image, self.right_image], axis=0)
            normalized_state = (self.robot_state - self.stats['qpos_mean']) / self.stats['qpos_std']

            # 转换为tensor
            model_dtype = next(self.policy.policy.parameters()).dtype
            curr_image = torch.from_numpy(images / 255.0).to(dtype=model_dtype).cuda()
            robot_state = torch.from_numpy(normalized_state).to(dtype=model_dtype).cuda().unsqueeze(0)

            # 处理输入批次
            batch = self.policy.process_batch_to_llava(
                curr_image, robot_state, self.task_description
            )

            # 运行推理
            # 🔧 重要：不设置固定随机种子！
            # 原因：固定seed会让扩散模型输出几乎不依赖输入变化
            # 扩散模型需要随机性来根据不同的视觉观察生成不同的动作
            with torch.inference_mode():
                all_actions = self.policy.policy(**batch, eval=True)

            # 后处理动作
            post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
            raw_actions = all_actions[0].cpu().numpy()  # (chunk_size, action_dim)

            # 应用后处理
            processed_actions = np.array([post_process(action) for action in raw_actions])

            # 发布动作序列
            action_msg = Float32MultiArray()
            action_msg.data = processed_actions.flatten().tolist()
            self.action_pub.publish(action_msg)

            # 更新统计
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time

            rospy.loginfo("推理完成 #%d: 动作序列形状=%s, 耗时=%.3fs",
                         self.inference_count,
                         processed_actions.shape,
                         inference_time)

        except Exception as e:
            rospy.logerr("推理错误: %s", e)
            import traceback
            rospy.logerr(traceback.format_exc())

    def print_stats(self, event):
        """打印性能统计"""
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count
            rospy.loginfo("=" * 60)
            rospy.loginfo("推理统计:")
            rospy.loginfo("  总推理次数: %d", self.inference_count)
            rospy.loginfo("  平均推理时间: %.3fs", avg_time)
            rospy.loginfo("  实际推理频率: %.2f Hz", 1.0 / avg_time if avg_time > 0 else 0)
            rospy.loginfo("=" * 60)

    def run(self):
        """运行服务器主循环"""
        rospy.loginfo("推理服务器开始运行...")
        rospy.spin()


def main():
    """主函数"""
    # 初始化日志系统
    logger = setup_logging()
    logger.info("="*60)
    logger.info("推理服务器启动中...")
    logger.info("="*60)

    # 模型配置
    action_head = 'droid_diffusion'
    policy_config = {
        "model_path": "/home/tianxiaoyan/TinyVLA/output/droid_multi_task_processed_latest",
        "model_base": "./checkpoints/llava-pythia-13b",
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
        "action_head_type": action_head,
    }

    # 任务描述 - 必须与训练数据一致！
    task_description = "pick up the wooden block and place it in the blue basket"

    logger.info(f"模型路径: {policy_config['model_path']}")
    logger.info(f"任务描述: {task_description}")

    try:
        # 创建并运行推理服务器
        server = InferenceServer(policy_config, task_description)
        server.run()

    except rospy.ROSInterruptException:
        logger.info("ROS中断")
        rospy.loginfo("ROS中断")
    except KeyboardInterrupt:
        logger.info("用户中断")
        rospy.loginfo("用户中断")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        rospy.logerr("服务器错误: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        rospy.logerr(traceback.format_exc())
    finally:
        logger.info("推理服务器已关闭")
        rospy.loginfo("推理服务器已关闭")


if __name__ == '__main__':
    main()
