"""
双机部署方案 - ROS分布式架构

机器A (推理服务器): 运行模型推理
机器B (控制主机): 运行机器人控制

网络配置:
- 机器A IP: 192.168.1.100
- 机器B IP: 192.168.1.101
- 网络延迟要求: <10ms
- 带宽要求: >100Mbps (推荐千兆网)
"""

# ============= 机器A: 推理服务器 =============
# inference_server.py

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import torch
import numpy as np
from eval_real_franka import llava_pythia_act_policy
import pickle

class InferenceServer:
    def __init__(self):
        rospy.init_node('tinyvla_inference_server', anonymous=False)

        # 加载模型
        print("加载TinyVLA模型...")
        self.policy_config = {
            "model_path": "/path/to/your/model",
            "model_base": "./checkpoints/llava-pythia-13b",
            "enable_lora": True,
            "conv_mode": "pythia",
            "action_head": "droid_diffusion",
            "action_head_type": "droid_diffusion",
        }
        self.policy = llava_pythia_act_policy(self.policy_config)

        # 加载统计信息
        stats_path = f"{self.policy_config['model_path']}/dataset_stats.pkl"
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # ROS通信
        self.bridge = CvBridge()

        # 订阅图像和状态
        self.image_left_sub = rospy.Subscriber(
            '/camera/left/image_raw', Image, self.image_left_callback
        )
        self.image_right_sub = rospy.Subscriber(
            '/camera/right/image_raw', Image, self.image_right_callback
        )
        self.state_sub = rospy.Subscriber(
            '/robot/state', Float32MultiArray, self.state_callback
        )

        # 发布动作
        self.action_pub = rospy.Publisher(
            '/inference/actions', Float32MultiArray, queue_size=1
        )

        # 缓存
        self.left_image = None
        self.right_image = None
        self.robot_state = None
        self.task_description = "pick up the white bowl"

        # 推理频率控制
        self.last_inference_time = 0
        self.inference_interval = 1.0  # 1Hz推理

        print("推理服务器启动完成")

    def image_left_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def image_right_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def state_callback(self, msg):
        self.robot_state = np.array(msg.data)

        # 检查是否可以进行推理
        current_time = rospy.get_time()
        if (current_time - self.last_inference_time) >= self.inference_interval:
            self.run_inference()
            self.last_inference_time = current_time

    def run_inference(self):
        """运行模型推理"""
        if self.left_image is None or self.right_image is None or self.robot_state is None:
            rospy.logwarn("等待数据...")
            return

        try:
            # 准备输入
            images = np.stack([self.left_image, self.right_image], axis=0)
            normalized_state = (self.robot_state - self.stats['qpos_mean']) / self.stats['qpos_std']

            # 转换为tensor
            model_dtype = next(self.policy.policy.parameters()).dtype
            curr_image = torch.from_numpy(images / 255.0).to(dtype=model_dtype).cuda()
            robot_state = torch.from_numpy(normalized_state).to(dtype=model_dtype).cuda().unsqueeze(0)

            # 推理
            batch = self.policy.process_batch_to_llava(
                curr_image, robot_state, self.task_description
            )

            with torch.inference_mode():
                all_actions = self.policy.policy(**batch, eval=True)

            # 后处理
            post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
            actions = all_actions[0].cpu().numpy()  # (chunk_size, action_dim)

            # 发布动作序列
            action_msg = Float32MultiArray()
            action_msg.data = actions.flatten().tolist()
            self.action_pub.publish(action_msg)

            rospy.loginfo(f"发布动作序列: chunk_size={actions.shape[0]}")

        except Exception as e:
            rospy.logerr(f"推理错误: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    server = InferenceServer()
    server.run()


# ============= 机器B: 控制主机 =============
# control_client.py

import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Float64
import numpy as np
import cv2
from collections import deque
from eval_real_franka import convert_actions

class ControlClient:
    def __init__(self):
        rospy.init_node('tinyvla_control_client', anonymous=False)

        # 相机
        self.left_cap = cv2.VideoCapture(4)
        self.right_cap = cv2.VideoCapture(10)

        # ROS发布
        self.image_left_pub = rospy.Publisher(
            '/camera/left/image_raw', Image, queue_size=1
        )
        self.image_right_pub = rospy.Publisher(
            '/camera/right/image_raw', Image, queue_size=1
        )
        self.state_pub = rospy.Publisher(
            '/robot/state', Float32MultiArray, queue_size=1
        )

        # 机器人控制
        self.pose_pub = rospy.Publisher(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped, queue_size=1
        )
        self.gripper_pub = rospy.Publisher(
            '/franka_gripper/goal_width', Float64, queue_size=1
        )

        # 订阅推理结果
        self.action_sub = rospy.Subscriber(
            '/inference/actions', Float32MultiArray, self.action_callback
        )

        # 订阅关节状态
        self.joint_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states',
            JointState, self.joint_state_callback
        )

        # 动作缓存
        self.action_buffer = deque(maxlen=100)
        self.current_action_idx = 0

        # 状态
        self.joint_positions = np.zeros(7)

        # 控制频率
        self.control_rate = rospy.Rate(50)  # 50Hz

        print("控制客户端启动完成")

    def joint_state_callback(self, msg):
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7])

    def action_callback(self, msg):
        """接收推理服务器的动作序列"""
        actions = np.array(msg.data).reshape(-1, 10)  # (chunk_size, 10)

        # 添加到缓存
        for action in actions:
            self.action_buffer.append(action)

        rospy.loginfo(f"接收动作序列: {len(actions)}步, 缓存大小: {len(self.action_buffer)}")

    def publish_sensor_data(self):
        """发布传感器数据到推理服务器"""
        # 采集图像
        ret_left, img_left = self.left_cap.read()
        ret_right, img_right = self.right_cap.read()

        if ret_left and ret_right:
            # 发布图像
            from cv_bridge import CvBridge
            bridge = CvBridge()

            msg_left = bridge.cv2_to_imgmsg(img_left, "rgb8")
            msg_right = bridge.cv2_to_imgmsg(img_right, "rgb8")

            self.image_left_pub.publish(msg_left)
            self.image_right_pub.publish(msg_right)

        # 发布机器人状态
        state_msg = Float32MultiArray()
        state_msg.data = self.joint_positions.tolist()
        self.state_pub.publish(state_msg)

    def execute_action(self):
        """执行一个动作"""
        if len(self.action_buffer) == 0:
            rospy.logwarn("动作缓存为空,等待推理...")
            return

        # 从缓存获取动作
        raw_action = self.action_buffer.popleft()

        # 转换动作
        action = convert_actions(raw_action, task_type="pick_up_bowl")

        # 发布位姿
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

        # 发布夹爪
        # 注意: 动作格式为 [x, y, z, rot_6d(6维), gripper(1维)] = 10维
        # 夹爪在索引9 (第10维)
        if len(action) >= 10:
            gripper_msg = Float64()
            gripper_msg.data = action[9]
            self.gripper_pub.publish(gripper_msg)

    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            # 发布传感器数据
            self.publish_sensor_data()

            # 执行动作
            self.execute_action()

            # 控制频率
            self.control_rate.sleep()

if __name__ == '__main__':
    client = ControlClient()
    client.run()
