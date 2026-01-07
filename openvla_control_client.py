#!/usr/bin/env python3
"""
OpenVLA 控制客户端 - 轻量级版本
在主机B上运行，不需要 transformers 等深度学习库

功能: 
1. 采集相机图像并发布
2. 接收 OpenVLA 推理结果
3. 控制 Franka 机械臂
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
import signal
import time
import logging
from datetime import datetime
import actionlib
from franka_gripper.msg import MoveGoal, MoveAction
from scipy.spatial.transform import Rotation as R


def setup_logging():
    """配置日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"openvla_control_{timestamp}.log"
    
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
    logger.info(f"日志文件:  {log_file}")
    return logger


class OpenVLAControlClient:
    def __init__(self, left_cam_id=4, right_cam_id=10, control_rate=5):
        """
        初始化 OpenVLA 控制客户端
        
        Args:
            left_cam_id: 左相机ID (OpenVLA 主要使用)
            right_cam_id:  右相机ID (保留兼容)
            control_rate: 控制频率(Hz)
        """
        rospy.init_node('openvla_control_client', anonymous=False)
        rospy.loginfo("正在初始化 OpenVLA 控制客户端...")
        
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
        
        # 控制频率
        self.control_rate = rospy.Rate(control_rate)
        
        # 上一个动作
        self.last_action = None
        self.last_gripper_width = None
        self.gripper_change_threshold = 0.005  # 5mm
        
        # 性能统计
        self.action_received_count = 0
        self. action_executed_count = 0
        
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
        
        # 夹爪 Action 客户端
        rospy.loginfo("正在初始化夹爪 Action 客户端...")
        self.gripper_client = actionlib.SimpleActionClient(
            '/franka_gripper/move',
            MoveAction
        )
        if not self.gripper_client. wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.logwarn("⚠️ 夹爪 Action 服务器未响应")
        else:
            rospy.loginfo("✅ 夹爪 Action 客户端已连接")
        
        # ROS订阅者 - 推理结果
        rospy. loginfo("正在设置ROS订阅者...")
        self.action_sub = rospy.Subscriber(
            '/inference/actions',
            Float32MultiArray,
            self. action_callback,
            queue_size=1
        )
        
        # ROS订阅者 - 关节状态
        self.joint_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states',
            JointState,
            self. joint_state_callback,
            queue_size=1
        )
        
        # ROS订阅者 - 末端执行器位姿
        self.ee_pose_sub = rospy.Subscriber(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            self.ee_pose_callback,
            queue_size=1
        )
        
        # 等待ROS连接建立
        rospy.sleep(1.0)
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("OpenVLA 控制客户端启动完成")
        rospy.loginfo(f"左相机ID: {left_cam_id} (主要)")
        rospy.loginfo(f"控制频率: {control_rate} Hz")
        rospy.loginfo("动作格式: 7-DoF 相对增量")
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
                    rospy.logerr(f"错误:  无法打开{name}相机 (ID: {cam_id})")
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    rospy.loginfo(f"{name}相机 (ID: {cam_id}) 初始化成功")
            
            rospy.loginfo("相机初始化完成")
            
        except Exception as e: 
            rospy.logerr(f"相机初始化错误: {e}")
    
    def joint_state_callback(self, msg):
        """接收关节状态"""
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[: 7])
    
    def ee_pose_callback(self, msg):
        """接收末端执行器位姿"""
        self.current_ee_pose = msg
    
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
            rospy.logwarn("还未接收到末端执行器位姿，返回默认姿态")
            return np. array([0.0, 1.0, 0.0, 0.0])  # 向下指向
        
        return np. array([
            self.current_ee_pose.pose.orientation. x,
            self.current_ee_pose.pose.orientation. y,
            self.current_ee_pose.pose.orientation. z,
            self.current_ee_pose.pose.orientation. w
        ])
    
    def action_callback(self, msg):
        """
        接收 OpenVLA 推理服务器的动作
        
        OpenVLA 输出:  [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        """
        try:
            action = np.array(msg.data)
            
            if len(action) != 7:
                rospy.logerr(f"无效的 OpenVLA 动作维度: {len(action)} (期望7)")
                return
            
            # 立即执行
            self.execute_openvla_action(action)
            self.action_received_count += 1
            
        except Exception as e:
            rospy.logerr(f"动作回调错误: {e}")
    
    def convert_openvla_action_to_pose(self, openvla_action):
        """
        将 OpenVLA 动作转换为 Franka 位姿控制
        
        Args: 
            openvla_action: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        
        Returns:
            pose_action: [x, y, z, qx, qy, qz, qw, gripper]
        """
        # 1. 获取当前状态
        current_pos = self.get_current_ee_position()
        current_quat = self.get_current_ee_quaternion()
        
        # 2. 提取增量
        delta_pos = openvla_action[: 3]
        delta_rot_euler = openvla_action[3: 6]  # 欧拉角增量 (rad)
        gripper = openvla_action[6]
        
        rospy.loginfo(f"OpenVLA 动作:  位移={delta_pos}, 旋转={delta_rot_euler}, 夹爪={gripper:. 3f}")
        
        # 3. 限制单步位移（安全）
        max_delta = 0.05  # 5cm
        delta_norm = np.linalg.norm(delta_pos)
        if delta_norm > max_delta: 
            rospy.logwarn(f"位移过大 ({delta_norm:.3f}m), 限制到 {max_delta}m")
            delta_pos = delta_pos / delta_norm * max_delta
        
        # 4. 计算目标位置 = 当前位置 + 增量
        target_pos = current_pos + delta_pos
        
        # 5. 限制到安全工作空间
        target_pos[0] = np.clip(target_pos[0], 0.1, 0.6)   # X
        target_pos[1] = np.clip(target_pos[1], -0.4, 0.4)  # Y
        target_pos[2] = np.clip(target_pos[2], 0.05, 0.7)  # Z
        
        rospy.loginfo(f"目标位置: {target_pos} (当前:  {current_pos})")
        
        # 6. 计算目标姿态
        delta_rot = R.from_euler('xyz', delta_rot_euler)
        current_rot = R.from_quat(current_quat)
        target_rot = current_rot * delta_rot
        target_quat = target_rot.as_quat()  # [x, y, z, w]
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        rospy.loginfo(f"目标四元数: {target_quat}")
        
        # 7. 处理夹爪
        gripper_width = gripper * 0.08
        gripper_width = np.clip(gripper_width, 0.0, 0.08)
        
        rospy.loginfo(f"夹爪宽度: {gripper_width:. 4f}m")
        
        # 8. 组合最终动作
        pose_action = np.concatenate([target_pos, target_quat, [gripper_width]])
        
        return pose_action
    
    def execute_openvla_action(self, openvla_action):
        """执行一个 OpenVLA 动作"""
        try:
            # 转换为 Franka 位姿
            pose_action = self.convert_openvla_action_to_pose(openvla_action)
            
            # 发布位姿
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "panda_link0"
            msg.pose. position.x = pose_action[0]
            msg.pose. position.y = pose_action[1]
            msg.pose. position.z = pose_action[2]
            msg.pose. orientation.x = pose_action[3]
            msg.pose. orientation.y = pose_action[4]
            msg.pose. orientation.z = pose_action[5]
            msg.pose. orientation.w = pose_action[6]
            self.pose_pub.publish(msg)
            
            # 发布夹爪控制
            gripper_width = pose_action[7]
            
            # 防抖动控制
            if self.last_gripper_width is not None: 
                width_change = abs(gripper_width - self.last_gripper_width)
                if width_change < self.gripper_change_threshold:
                    rospy.logdebug(f"夹爪变化过小 ({width_change:.4f}m), 跳过")
                    self.action_executed_count += 1
                    self.last_action = pose_action
                    return
            
            # 发送夹爪命令
            self.control_gripper(gripper_width, speed=0.1, wait=False)
            self.last_gripper_width = gripper_width
            
            # 更新统计
            self.action_executed_count += 1
            self.last_action = pose_action
            
            rospy.logdebug(f"动作已执行: 位置={pose_action[:3]}, 夹爪={gripper_width:.4f}")
            
        except Exception as e:
            rospy.logerr(f"动作执行错误: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def control_gripper(self, width, speed=0.1, wait=False, timeout=2.0):
        """控制夹爪"""
        try:
            width = np.clip(width, 0.0, 0.08)
            
            # 取消之前的命令
            self.gripper_client.cancel_all_goals()
            
            # 创建 Goal
            goal = MoveGoal()
            goal.width = width
            goal.speed = speed
            
            # 发送 Goal
            self.gripper_client.send_goal(goal)
            
            if wait:
                finished = self.gripper_client. wait_for_result(rospy.Duration(timeout))
                if finished:
                    rospy.loginfo("✅ 夹爪动作完成")
                else: 
                    rospy.logwarn(f"⚠️ 夹爪动作超时 ({timeout}s)")
            else:
                rospy.logdebug(f"✅ 夹爪命令已发送:  {width:.4f}m (异步)")
                
        except Exception as e:
            rospy. logerr(f"夹爪控制错误: {e}")
    
    def publish_sensor_data(self):
        """发布传感器数据到推理服务器"""
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
            state_msg.data = self.joint_positions. tolist()
            self.state_pub.publish(state_msg)
            
        except Exception as e:
            rospy.logerr(f"传感器数据发布错误: {e}")
    
    def emergency_stop(self):
        """紧急停止"""
        rospy.logwarn("执行紧急停止!")
        
        msg = PoseStamped()
        msg.header.stamp = rospy. Time.now()
        msg.header.frame_id = "panda_link0"
        msg.pose.position.x = 0.3
        msg.pose.position. y = 0.0
        msg.pose.position.z = 0.3
        msg. pose.orientation.x = 0.0
        msg.pose. orientation.y = 1.0
        msg.pose. orientation.z = 0.0
        msg.pose.orientation. w = 0.0
        self.pose_pub.publish(msg)
        
        self.control_gripper(0.08, speed=0.1)
        rospy.loginfo("紧急停止命令已发送")
    
    def run(self):
        """运行控制客户端主循环"""
        rospy. loginfo("OpenVLA 控制客户端开始运行...")
        
        try:
            while not rospy. is_shutdown():
                # 发布传感器数据
                self.publish_sensor_data()
                
                # 控制频率
                self.control_rate. sleep()
                
        except rospy.ROSInterruptException:
            rospy.loginfo("ROS中断")
        except KeyboardInterrupt:
            rospy.loginfo("用户中断")
        finally:
            rospy.loginfo(f"总接收动作:  {self.action_received_count}")
            rospy.loginfo(f"总执行动作: {self.action_executed_count}")
            self.emergency_stop()
    
    def __del__(self):
        """清理资源"""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap. release()


def signal_handler(sig, frame):
    """信号处理器"""
    rospy.loginfo("接收到中断信号,正在安全关闭...")
    rospy.signal_shutdown("用户中断")


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("OpenVLA 控制客户端启动中...")
    logger.info("=" * 60)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        client = OpenVLAControlClient(
            left_cam_id=4,   # 修改为实际的相机ID
            right_cam_id=10,  # 修改为实际的相机ID
            control_rate=5    # 5Hz 控制频率
        )
        client.run()
        
    except Exception as e: 
        logger.error(f"客户端错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("OpenVLA 控制客户端已关闭")


if __name__ == '__main__':
    main()
