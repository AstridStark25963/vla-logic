#!/usr/bin/env python3
"""
控制客户端 - 修正版（使用正确的夹爪接口）
在主机B上运行

修正内容：
1. 使用 actionlib 调用 /franka_gripper/move Action
2. 不再使用不存在的 /franka_gripper/goal_width topic

功能:
1. 采集相机图像并发布到推理服务器
2. 发布机器人状态到推理服务器
3. 接收推理服务器的动作序列
4. 执行动作控制Franka机械臂
"""

import os
import sys
import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque
import signal
import time
import logging
from datetime import datetime
import actionlib
from franka_gripper.msg import MoveGoal, MoveAction
import threading

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_utils_fixed import convert_actions
# 🔬 导入诊断工具
from convert_actions_diagnostic import ActionConverter


def setup_logging():
    """配置日志系统：同时输出到终端和文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"control_client_{timestamp}.log"

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")

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

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fileobj = open(log_file, 'a', encoding='utf-8')
    sys.stdout = TeeOutput(original_stdout, log_fileobj)
    sys.stderr = TeeOutput(original_stderr, log_fileobj)

    return logger


class ControlClient:
    def __init__(self, left_cam_id=4, right_cam_id=10, control_rate=5):
        """
        初始化控制客户端

        Args:
            left_cam_id: 左相机ID
            right_cam_id: 右相机ID
            control_rate: 控制频率(Hz) - ⚠️ 必须 ≤ 5Hz，防止机械臂乱飞
        """
        # 初始化ROS节点
        rospy.init_node('tinyvla_control_client', anonymous=False)
        rospy.loginfo("正在初始化控制客户端...")

        # 相机配置
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.left_cap = None
        self.right_cap = None

        # 初始化相机
        self.init_cameras()

        # CV Bridge
        self.bridge = CvBridge()

        # 机器人状态
        self.joint_positions = np.zeros(7)
        self.current_ee_pose = None

        # 动作缓存
        self.action_buffer = deque(maxlen=100)
        self.last_action = None

        # 夹爪状态追踪
        self.last_gripper_width = None  # 上次发送的夹爪宽度
        self.gripper_change_threshold = 0.005  # 变化阈值(m) - 小于此值不发送新命令 (降低到5mm)

        # 控制频率
        self.control_rate = rospy.Rate(control_rate)
        self.sensor_publish_rate = rospy.Rate(10)

        # 🔧 线程控制标志
        self.sensor_thread = None
        self.running = False

        # 性能统计
        self.action_received_count = 0
        self.action_executed_count = 0
        self.last_stats_time = time.time()

        # ROS发布者 - 传感器数据
        rospy.loginfo("正在设置ROS发布者...")
        self.image_left_pub = rospy.Publisher(
            '/camera/left/image_raw',
            Image,
            queue_size=1
        )
        self.image_right_pub = rospy.Publisher(
            '/camera/right/image_raw',
            Image,
            queue_size=1
        )
        self.state_pub = rospy.Publisher(
            '/robot/state',
            Float32MultiArray,
            queue_size=1
        )

        # ROS发布者 - 机器人控制
        self.pose_pub = rospy.Publisher(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=1
        )

        # ✅ 修正：使用 Action 客户端控制夹爪
        rospy.loginfo("正在初始化夹爪 Action 客户端...")
        self.gripper_client = actionlib.SimpleActionClient(
            '/franka_gripper/move',
            MoveAction
        )
        rospy.loginfo("等待夹爪 Action 服务器...")
        if not self.gripper_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.logwarn("⚠️ 夹爪 Action 服务器未响应，夹爪控制可能不可用")
        else:
            rospy.loginfo("✅ 夹爪 Action 客户端已连接")

        # ROS订阅者 - 推理结果
        rospy.loginfo("正在设置ROS订阅者...")
        self.action_sub = rospy.Subscriber(
            '/inference/actions',
            Float32MultiArray,
            self.action_callback,
            queue_size=1
        )

        # ROS订阅者 - 关节状态
        self.joint_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states',
            JointState,
            self.joint_state_callback,
            queue_size=1
        )

        # ROS订阅者 - 末端执行器位姿
        self.ee_pose_sub = rospy.Subscriber(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            self.ee_pose_callback,
            queue_size=1
        )

        # 性能监控定时器
        rospy.Timer(rospy.Duration(10.0), self.print_stats)

        # 等待ROS连接建立
        rospy.sleep(1.0)

        rospy.loginfo("=" * 60)
        rospy.loginfo("控制客户端启动完成")
        rospy.loginfo("夹爪接口: /franka_gripper/move (Action)")
        rospy.loginfo("左相机ID: %d", left_cam_id)
        rospy.loginfo("右相机ID: %d", right_cam_id)
        rospy.loginfo("控制频率: %d Hz", control_rate)

        # 🔬 诊断工具初始化
        rospy.loginfo("=" * 60)
        rospy.loginfo("🔬 诊断模式配置:")

        # ⚙️ 实验开关 - 根据需要修改这些值
        EXPERIMENT_LOCK_ROTATION = False      # 实验A: True=锁死姿态（验证旋转问题）
        EXPERIMENT_LOCK_TRANSLATION = False   # 实验B: True=锁死平移（验证姿态稳定性）
        EXPERIMENT_USE_EE_FRAME = False       # 实验C: True=使用EE frame delta（验证坐标系）
        EXPERIMENT_FORCE_NORMALIZE_6D = True  # 强制normalize 6D rotation（推荐True）

        rospy.loginfo("  锁死姿态: %s", EXPERIMENT_LOCK_ROTATION)
        rospy.loginfo("  锁死平移: %s", EXPERIMENT_LOCK_TRANSLATION)
        rospy.loginfo("  EE Frame Delta: %s", EXPERIMENT_USE_EE_FRAME)
        rospy.loginfo("  强制Normalize 6D: %s", EXPERIMENT_FORCE_NORMALIZE_6D)

        self.action_converter = ActionConverter(
            lock_rotation=EXPERIMENT_LOCK_ROTATION,
            lock_translation=EXPERIMENT_LOCK_TRANSLATION,
            force_normalize_6d=EXPERIMENT_FORCE_NORMALIZE_6D,
            use_ee_frame_delta=EXPERIMENT_USE_EE_FRAME,
            verbose_diagnostics=True  # 输出详细日志
        )
        rospy.loginfo("✅ 诊断转换器已初始化")
        rospy.loginfo("=" * 60)

        # 轨迹记录初始化
        self.trajectory_data = {
            'actions': [],
            'ee_positions': [],
            'joint_positions': [],
            'raw_actions': [],
            'timestamps': []
        }
        self.trajectory_recording = True
        self.trajectory_save_interval = 20
        self.trajectory_step_count = 0
        rospy.loginfo("✅ 轨迹记录已初始化")
        rospy.loginfo("=" * 60)

    def init_cameras(self):
        """初始化相机"""
        rospy.loginfo("正在初始化相机...")

        try:
            self.left_cap = cv2.VideoCapture(self.left_cam_id)
            self.right_cap = cv2.VideoCapture(self.right_cam_id)

            for cap, name, cam_id in [(self.left_cap, "左", self.left_cam_id),
                                       (self.right_cap, "右", self.right_cam_id)]:
                if not cap.isOpened():
                    rospy.logerr("错误: 无法打开%s相机 (ID: %d)", name, cam_id)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    rospy.loginfo("%s相机 (ID: %d) 初始化成功", name, cam_id)

            rospy.loginfo("相机初始化完成")

        except Exception as e:
            rospy.logerr("相机初始化错误: %s", e)

    def joint_state_callback(self, msg):
        """接收关节状态"""
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7])
            rospy.logdebug("关节状态: %s", self.joint_positions)

    def ee_pose_callback(self, msg):
        """接收末端执行器位姿"""
        self.current_ee_pose = msg
        rospy.logdebug("末端执行器位姿: x=%.3f, y=%.3f, z=%.3f",
                      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def get_current_ee_position(self):
        """获取当前末端执行器位置"""
        if self.current_ee_pose is None:
            rospy.logwarn("还未接收到末端执行器位姿，使用默认位置")
            return np.array([0.3, 0.0, 0.3])

        return np.array([
            self.current_ee_pose.pose.position.x,
            self.current_ee_pose.pose.position.y,
            self.current_ee_pose.pose.position.z
        ])

    def get_current_ee_quaternion(self):
        """获取当前末端执行器四元数 [x, y, z, w]"""
        if self.current_ee_pose is None:
            rospy.logwarn("还未接收到末端执行器位姿，返回None")
            return None

        return np.array([
            self.current_ee_pose.pose.orientation.x,
            self.current_ee_pose.pose.orientation.y,
            self.current_ee_pose.pose.orientation.z,
            self.current_ee_pose.pose.orientation.w
        ])

    def action_callback(self, msg):
        """接收推理服务器的动作序列"""
        try:
            actions = np.array(msg.data)

            # 确定动作维度
            if len(actions) % 10 == 0:
                action_dim = 10
            elif len(actions) % 8 == 0:
                action_dim = 8
            else:
                rospy.logerr("无效的动作维度: %d", len(actions))
                return

            actions = actions.reshape(-1, action_dim)

            # 添加到缓存
            for action in actions:
                self.action_buffer.append(action)

            self.action_received_count += len(actions)

            rospy.loginfo("接收动作序列: %d步, 缓存大小: %d",
                         len(actions), len(self.action_buffer))

        except Exception as e:
            rospy.logerr("动作回调错误: %s", e)

    def publish_sensor_data(self):
        """发布传感器数据到推理服务器（单次调用）"""
        try:
            # 采集图像
            ret_left, img_left = self.left_cap.read() if self.left_cap else (False, None)
            ret_right, img_right = self.right_cap.read() if self.right_cap else (False, None)

            # 发布图像
            if ret_left and img_left is not None:
                img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                msg_left = self.bridge.cv2_to_imgmsg(img_left_rgb, "rgb8")
                msg_left.header.stamp = rospy.Time.now()
                self.image_left_pub.publish(msg_left)

            if ret_right and img_right is not None:
                img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                msg_right = self.bridge.cv2_to_imgmsg(img_right_rgb, "rgb8")
                msg_right.header.stamp = rospy.Time.now()
                self.image_right_pub.publish(msg_right)

            # 发布机器人状态
            state_msg = Float32MultiArray()
            state_msg.data = self.joint_positions.tolist()
            self.state_pub.publish(state_msg)

            rospy.logdebug("传感器数据已发布")

        except Exception as e:
            rospy.logerr("传感器数据发布错误: %s", e)

    def sensor_publishing_loop(self):
        """
        🔧 传感器发布线程循环
        在单独的线程中以10Hz频率发布传感器数据
        """
        rospy.loginfo("✅ 传感器发布线程已启动 (10 Hz)")
        rate = rospy.Rate(10)  # 10Hz - 推理服务器不需要太高频率

        while self.running and not rospy.is_shutdown():
            try:
                self.publish_sensor_data()
                rate.sleep()
            except Exception as e:
                rospy.logerr("传感器发布线程错误: %s", e)

        rospy.loginfo("传感器发布线程已停止")

    def control_gripper(self, width, speed=0.1, wait=False, timeout=2.0):
        """
        ✅ 修正：使用 Action 接口控制夹爪 + 防抖动控制 + 可选等待

        Args:
            width: 夹爪宽度 (m)，范围 [0.0, 0.08]
            speed: 夹爪速度 (m/s)
            wait: 是否等待夹爪完成动作 (默认False，异步执行)
            timeout: 等待超时时间 (秒)
        """
        try:
            width = np.clip(width, 0.0, 0.08)

            # 🔧 防抖动：只有当夹爪变化超过阈值时才发送命令
            if self.last_gripper_width is not None:
                width_change = abs(width - self.last_gripper_width)
                if width_change < self.gripper_change_threshold:
                    rospy.logdebug("夹爪变化过小 (%.4f < %.4f)，跳过命令",
                                   width_change, self.gripper_change_threshold)
                    return

            # 取消之前的夹爪命令（防止冲突）
            self.gripper_client.cancel_all_goals()

            # 创建 Move Goal
            goal = MoveGoal()
            goal.width = width
            goal.speed = speed

            # 发送 Goal
            self.gripper_client.send_goal(goal)

            # 更新记录
            change_amount = width - (self.last_gripper_width or width)
            self.last_gripper_width = width

            # 🔧 如果需要等待夹爪完成
            if wait:
                rospy.loginfo("⏳ 夹爪命令已发送: %.4f m (变化: %.4f m)，等待完成...",
                             width, change_amount)
                # 等待夹爪完成，带超时
                finished = self.gripper_client.wait_for_result(rospy.Duration(timeout))
                if finished:
                    rospy.loginfo("✅ 夹爪动作完成")
                else:
                    rospy.logwarn("⚠️ 夹爪动作超时 (%.1fs)", timeout)
            else:
                rospy.loginfo("✅ 夹爪命令已发送: %.4f m (变化: %.4f m，异步执行)",
                             width, change_amount)

        except Exception as e:
            rospy.logerr("夹爪控制错误: %s", e)

    def execute_action(self):
        """执行一个动作"""
        if len(self.action_buffer) == 0:
            rospy.logdebug("动作缓存为空,等待推理...")
            return

        try:
            # 🔧 改进策略：如果缓存太多（>15），只取最新的1个动作，清理旧的
            # 如果缓存适中（1-15），正常执行
            if len(self.action_buffer) > 15:
                rospy.loginfo_throttle(5.0, "缓存过大(%d)，清理旧动作", len(self.action_buffer))
                # 保留最新的5个动作，清除其余的
                while len(self.action_buffer) > 5:
                    self.action_buffer.popleft()  # 从头部删除旧动作

            # 取最新的动作执行
            raw_action = self.action_buffer.pop()

            smoothing_factor = 0.05
            current_ee_pos = self.get_current_ee_position()

            rospy.loginfo_throttle(5.0, "当前末端执行器位置: [%.3f, %.3f, %.3f]",
                                  current_ee_pos[0], current_ee_pos[1], current_ee_pos[2])

            # 🔬 使用诊断版转换动作
            current_ee_quat = self.get_current_ee_quaternion()

            action = self.action_converter.convert(
                pred_action=raw_action,
                current_ee_pos=current_ee_pos,
                current_ee_quat=current_ee_quat,
                last_action=self.last_action,
                smoothing_factor=smoothing_factor
            )
            # 注意: action现在是8维 [xyz(3), quat(4), gripper(1)]

            # 发布笛卡尔位姿
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "panda_link0"
            msg.pose.position.x = action[0]
            msg.pose.position.y = action[1]
            msg.pose.position.z = action[2]
            msg.pose.orientation.x = action[3]
            msg.pose.orientation.y = action[4]
            msg.pose.orientation.z = action[5]
            msg.pose.orientation.w = action[6]
            self.pose_pub.publish(msg)

            # ✅ 修正：使用 Action 接口发布夹爪控制
            # 注意: convert_actions 返回 [x, y, z, quat(4维), gripper(1维)] = 8维
            # 夹爪在索引7 (第8维)
            if len(action) >= 8:
                gripper_width = np.clip(action[7], 0.0, 0.08)

                # 🔧 位置同步控制：放宽误差要求,允许更灵活的夹爪控制
                target_pos = np.array([action[0], action[1], action[2]])
                current_pos = self.get_current_ee_position()
                position_error = np.linalg.norm(target_pos - current_pos)

                # 位置误差阈值：100mm (放宽要求,更容易触发夹爪控制)
                position_threshold = 0.10
                arm_near_target = position_error < position_threshold

                # 🔧 启发式控制: 当末端执行器接近木块且高度很低时,强制关闭夹爪
                target_x = action[0]
                target_y = action[1]
                target_z = action[2]

                # 木块区域判断 (基于实际位置):
                # 小木块实际位置: x=0.467, y=-0.041, z=0.024
                # X ∈ [0.40, 0.53] (以0.467为中心，±6.7cm)
                # Y ∈ [-0.08, 0.00] (以-0.041为中心，±4cm)
                # Z < 0.08 (桌面2.4cm + 5.6cm余量)
                in_grasp_region = (0.40 < target_x < 0.53 and
                                   -0.08 < target_y < 0.00 and
                                   target_z < 0.08 and
                                   position_error < 0.05)  # 且距离目标<5cm

                # 注意: 在抓取区域且高度低时强制关闭夹爪
                is_closing_to_grasp = False
                if in_grasp_region and gripper_width > 0.035:
                    rospy.loginfo("🔧 启发式控制: 到达抓取位置 (x=%.3f, y=%.3f, z=%.3f, 误差=%.3fm), 强制关闭夹爪",
                                  target_x, target_y, target_z, position_error)
                    gripper_width = 0.020  # 强制关闭到抓取宽度 (2.0cm) - 更紧地夹住木块
                    is_closing_to_grasp = True  # 标记为关键抓取动作

                # 发送夹爪命令的条件 (显著放宽):
                # 1. 机械臂接近目标位置 (误差 < 10cm) - 或 -
                # 2. 在抓取区域且接近 (< 5cm)
                if arm_near_target or in_grasp_region:
                    # 🔧 判断是否需要等待夹爪完成
                    # 在以下情况等待：1) 关键抓取动作  2) 夹爪宽度变化较大 (>2cm)
                    gripper_change = abs(gripper_width - (self.last_gripper_width or gripper_width))
                    should_wait = is_closing_to_grasp or gripper_change > 0.02

                    # 关键抓取时等待，但缩短超时时间避免阻塞太久
                    # 其他情况不等待，异步执行以保持高频率
                    if is_closing_to_grasp:
                        self.control_gripper(gripper_width, speed=0.1, wait=True, timeout=0.5)
                    else:
                        self.control_gripper(gripper_width, speed=0.1, wait=False)
                else:
                    rospy.logdebug("⏸️ 等待机械臂到位 (误差=%.3fm), 暂不改变夹爪", position_error)


            # 更新记录
            self.last_action = action
            self.action_executed_count += 1

            # 记录轨迹数据
            if self.trajectory_recording:
                try:
                    self.trajectory_data['raw_actions'].append(raw_action.copy())
                    self.trajectory_data['actions'].append(action.copy())
                    self.trajectory_data['ee_positions'].append(current_ee_pos.copy())
                    self.trajectory_data['joint_positions'].append(self.joint_positions.copy())
                    self.trajectory_data['timestamps'].append(time.time())

                    self.trajectory_step_count += 1

                    if self.trajectory_step_count % self.trajectory_save_interval == 0:
                        self.save_trajectory(auto_save=True)
                        rospy.loginfo_throttle(10.0, "轨迹自动保存: %d 步", self.trajectory_step_count)

                except Exception as e:
                    rospy.logerr("轨迹记录错误: %s", e)

            rospy.logdebug("动作已执行: 位置=(%.3f, %.3f, %.3f), 夹爪=%.4f",
                          action[0], action[1], action[2], action[7] if len(action) >= 8 else 0.0)

            # 移除sleep，让循环以最快速度执行（由control_rate控制）
            # time.sleep(0.02)

        except Exception as e:
            rospy.logerr("动作执行错误: %s", e)

    def emergency_stop(self):
        """紧急停止"""
        rospy.logwarn("执行紧急停止!")

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"
        msg.pose.position.x = 0.3
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.3
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

        # ✅ 修正：使用 Action 接口打开夹爪
        self.control_gripper(0.08, speed=0.1)

        rospy.loginfo("紧急停止命令已发送")

    def save_trajectory(self, auto_save=False):
        """保存轨迹数据"""
        import pickle

        if len(self.trajectory_data['ee_positions']) == 0:
            rospy.logwarn("轨迹数据为空，跳过保存")
            return

        if auto_save:
            filename = 'real_trajectory_temp.pkl'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'real_trajectory_{timestamp}.pkl'

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectory_data, f)

            ee_positions = np.array(self.trajectory_data['ee_positions'])
            rospy.loginfo("")
            rospy.loginfo("="*60)
            rospy.loginfo("✅ 真实轨迹已保存到: %s", filename)
            rospy.loginfo("   包含 %d 个时间步", len(self.trajectory_data['ee_positions']))
            rospy.loginfo("   末端执行器位置范围:")
            rospy.loginfo("     X: [%.3f, %.3f] m", ee_positions[:, 0].min(), ee_positions[:, 0].max())
            rospy.loginfo("     Y: [%.3f, %.3f] m", ee_positions[:, 1].min(), ee_positions[:, 1].max())
            rospy.loginfo("     Z: [%.3f, %.3f] m", ee_positions[:, 2].min(), ee_positions[:, 2].max())
            rospy.loginfo("="*60)
            rospy.loginfo("")

        except Exception as e:
            rospy.logerr("保存轨迹失败: %s", e)

    def print_stats(self, event):
        """打印性能统计"""
        current_time = time.time()
        elapsed_time = current_time - self.last_stats_time

        if elapsed_time > 0:
            receive_rate = self.action_received_count / elapsed_time
            execute_rate = self.action_executed_count / elapsed_time

            rospy.loginfo("=" * 60)
            rospy.loginfo("控制统计:")
            rospy.loginfo("  接收动作数: %d (%.2f Hz)",
                         self.action_received_count, receive_rate)
            rospy.loginfo("  执行动作数: %d (%.2f Hz)",
                         self.action_executed_count, execute_rate)
            rospy.loginfo("  缓存大小: %d", len(self.action_buffer))
            rospy.loginfo("=" * 60)

            self.action_received_count = 0
            self.action_executed_count = 0
            self.last_stats_time = current_time

    def run(self):
        """运行控制客户端主循环"""
        rospy.loginfo("控制客户端开始运行...")

        # 🔧 启动传感器发布线程
        self.running = True
        self.sensor_thread = threading.Thread(target=self.sensor_publishing_loop, daemon=True)
        self.sensor_thread.start()
        rospy.loginfo("✅ 传感器发布线程已创建（独立运行，不阻塞控制循环）")

        try:
            while not rospy.is_shutdown():
                # 🚀 主循环只执行动作，不再发布传感器数据
                # 传感器数据由独立线程以10Hz频率发布
                self.execute_action()
                self.control_rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("ROS中断")
        except KeyboardInterrupt:
            rospy.loginfo("用户中断")
        finally:
            # 停止传感器线程
            self.running = False
            if self.sensor_thread and self.sensor_thread.is_alive():
                self.sensor_thread.join(timeout=2.0)
                rospy.loginfo("传感器发布线程已停止")

            # 🔬 打印诊断统计
            rospy.loginfo("")
            rospy.loginfo("="*60)
            rospy.loginfo("🔬 正在生成诊断报告...")
            rospy.loginfo("="*60)
            self.action_converter.print_statistics()

            rospy.loginfo("正在保存轨迹数据...")
            self.save_trajectory(auto_save=False)
            self.emergency_stop()

    def __del__(self):
        """清理资源"""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
        rospy.loginfo("相机资源已释放")


def signal_handler(sig, frame):
    """信号处理器"""
    rospy.loginfo("接收到中断信号,正在安全关闭...")
    rospy.signal_shutdown("用户中断")


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("="*60)
    logger.info("控制客户端启动中（修正版 - 使用 Action 接口）...")
    logger.info("="*60)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        client = ControlClient(left_cam_id=4, right_cam_id=10, control_rate=5)
        client.run()

    except rospy.ROSInterruptException:
        logger.info("ROS中断")
        rospy.loginfo("ROS中断")
    except KeyboardInterrupt:
        logger.info("用户中断")
        rospy.loginfo("用户中断")
    except Exception as e:
        logger.error(f"客户端错误: {e}")
        rospy.logerr("客户端错误: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        rospy.logerr(traceback.format_exc())
    finally:
        logger.info("控制客户端已关闭")
        rospy.loginfo("控制客户端已关闭")


if __name__ == '__main__':
    main()
