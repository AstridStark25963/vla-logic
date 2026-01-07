#!/usr/bin/env python3
"""
OpenVLA Client 环境检查工具
检查所有依赖和配置是否正确
"""

import os
import sys


def check_ros_environment():
    """检查 ROS 环境"""
    print("\n[1/6] 检查 ROS 环境...")
    
    ros_master_uri = os.environ.get('ROS_MASTER_URI')
    ros_ip = os. environ.get('ROS_IP')
    
    if ros_master_uri: 
        print(f"  ✓ ROS_MASTER_URI: {ros_master_uri}")
    else:
        print(f"  ✗ ROS_MASTER_URI 未设置")
        return False
    
    if ros_ip:
        print(f"  ✓ ROS_IP: {ros_ip}")
    else:
        print(f"  ✗ ROS_IP 未设置")
        return False
    
    return True


def check_ros_connection():
    """检查 ROS Master 连接"""
    print("\n[2/6] 检查 ROS Master 连接...")
    
    try:
        import rospy
        import subprocess
        
        # 尝试列出 topics
        result = subprocess.run(
            ['timeout', '3', 'rostopic', 'list'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✓ ROS Master 可访问")
            return True
        else:
            print("  ⚠️ 无法连接到 ROS Master")
            print("  提示: 确保主机B上 roscore 已启动")
            return False
            
    except Exception as e:
        print(f"  ✗ ROS 连接检查失败: {e}")
        return False


def check_cameras(left_id=4, right_id=10):
    """检查相机设备"""
    print(f"\n[3/6] 检查相机设备...")
    
    left_path = f"/dev/video{left_id}"
    right_path = f"/dev/video{right_id}"
    
    left_ok = os.path.exists(left_path)
    right_ok = os.path.exists(right_path)
    
    if left_ok: 
        print(f"  ✓ 左相机 {left_path} 存在")
    else:
        print(f"  ✗ 左相机 {left_path} 不存在")
    
    if right_ok: 
        print(f"  ✓ 右相机 {right_path} 存在")
    else:
        print(f"  ⚠️ 右相机 {right_path} 不存在 (可选)")
    
    return left_ok


def check_python_dependencies():
    """检查 Python 依赖"""
    print("\n[4/6] 检查 Python 依赖...")
    
    dependencies = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'scipy': 'scipy',
        'rospy': 'rospy (ROS)',
        'sensor_msgs': 'sensor_msgs (ROS)',
        'geometry_msgs': 'geometry_msgs (ROS)',
        'actionlib': 'actionlib (ROS)',
    }
    
    all_ok = True
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (运行:  pip install {package. split()[0]})")
            all_ok = False
    
    return all_ok


def check_franka_topics():
    """检查 Franka 相关的 ROS Topics"""
    print("\n[5/6] 检查 Franka 相��� Topics...")
    
    try:
        import subprocess
        
        result = subprocess.run(
            ['timeout', '3', 'rostopic', 'list'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("  ⚠️ 无法获取 Topic 列表")
            return False
        
        topics = result.stdout.split('\n')
        
        required_topics = [
            '/franka_state_controller/joint_states',
            '/cartesian_impedance_example_controller/equilibrium_pose',
        ]
        
        optional_topics = [
            '/franka_gripper/move',
        ]
        
        all_ok = True
        
        for topic in required_topics: 
            if topic in topics:
                print(f"  ✓ {topic}")
            else:
                print(f"  ✗ {topic} (Franka 控制器可能未启动)")
                all_ok = False
        
        for topic in optional_topics:
            if topic in topics:
                print(f"  ✓ {topic}")
            else:
                print(f"  ⚠️ {topic} (可选，夹爪控制)")
        
        return all_ok
        
    except Exception as e: 
        print(f"  ✗ Topic 检查失败: {e}")
        return False


def check_network():
    """检查网络连通性"""
    print("\n[6/6] 检查网络连通性...")
    
    ros_master_uri = os.environ.get('ROS_MASTER_URI', '')
    
    if not ros_master_uri:
        print("  ⚠️ ROS_MASTER_URI 未设置")
        return False
    
    # 提取主机地址
    try:
        import urllib.parse
        parsed = urllib. parse.urlparse(ros_master_uri)
        host = parsed.hostname
        
        if host:
            import subprocess
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '2', host],
                capture_output=True
            )
            
            if result.returncode == 0:
                print(f"  ✓ 可以 ping 通 {host}")
                return True
            else:
                print(f"  ✗ 无法 ping 通 {host}")
                return False
    except Exception as e:
        print(f"  ⚠️ 网络检查失败: {e}")
        return False


def print_summary(results):
    """打印检查结果摘要"""
    print("\n" + "=" * 60)
    print("环境检查摘要")
    print("=" * 60)
    
    checks = [
        ("ROS 环境配置", results['ros_env']),
        ("ROS Master 连接", results['ros_connection']),
        ("相机设备", results['cameras']),
        ("Python 依赖", results['python_deps']),
        ("Franka Topics", results['franka_topics']),
        ("网络连通性", results['network']),
    ]
    
    all_passed = True
    
    for name, passed in checks: 
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name: 20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ 所有检查通过！可以启动 Client。")
        return 0
    else:
        print("\n⚠️ 部分检查未通过，请修复后再启动。")
        return 1


def main():
    """主函数"""
    print("=" * 60)
    print("OpenVLA Client 环境检查工具")
    print("=" * 60)
    
    # 读取配置（如果有命令行参数）
    left_cam_id = 4
    right_cam_id = 10
    
    if len(sys.argv) >= 2:
        left_cam_id = int(sys.argv[1])
    if len(sys.argv) >= 3:
        right_cam_id = int(sys.argv[2])
    
    # 执行检查
    results = {
        'ros_env': check_ros_environment(),
        'ros_connection': check_ros_connection(),
        'cameras': check_cameras(left_cam_id, right_cam_id),
        'python_deps': check_python_dependencies(),
        'franka_topics': check_franka_topics(),
        'network': check_network(),
    }
    
    # 打印摘要
    exit_code = print_summary(results)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
