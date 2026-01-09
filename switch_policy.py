#!/usr/bin/env python3
"""
切换策略模型工具

快速切换 TinyVLA 和 OpenVLA-7B 模型
"""

import sys
import os


def update_inference_server(policy_type):
    """
    更新 inference_server.py 中的策略类型
    
    Args:
        policy_type: 'tinyvla' 或 'openvla'
    """
    filepath = os.path.join(os.path.dirname(__file__), 'inference_server.py')
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换 USE_POLICY 行
    import re
    
    # 匹配 USE_POLICY = 'xxx' 这一行
    pattern = r"USE_POLICY\s*=\s*['\"](\w+)['\"]"
    match = re.search(pattern, content)
    
    if match:
        old_policy = match.group(1)
        if old_policy == policy_type:
            print(f"✅ 已经是 {policy_type} 模型，无需修改")
            return True
        
        # 替换
        new_content = re.sub(
            pattern,
            f"USE_POLICY = '{policy_type}'",
            content
        )
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 成功切换: {old_policy} -> {policy_type}")
        return True
    else:
        print(f"❌ 未找到 USE_POLICY 配置行")
        return False


def show_current_policy():
    """显示当前使用的策略"""
    filepath = os.path.join(os.path.dirname(__file__), 'inference_server.py')
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    pattern = r"USE_POLICY\s*=\s*['\"](\w+)['\"]"
    match = re.search(pattern, content)
    
    if match:
        policy = match.group(1)
        return policy
    else:
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("VLA 策略模型切换工具")
    print("=" * 60)
    print()
    
    # 显示当前策略
    current = show_current_policy()
    if current:
        print(f"📍 当前策略: {current.upper()}")
        
        if current == 'openvla':
            print("   OpenVLA-7B (DINOv2 + SigLIP, Llama-2-7b)")
        elif current == 'tinyvla':
            print("   TinyVLA (CLIP ViT, Pythia-1.3B)")
    else:
        print("⚠️  无法确定当前策略")
    
    print()
    print("可用选项:")
    print("  1. OpenVLA-7B (openvla)")
    print("  2. TinyVLA (tinyvla)")
    print("  3. 退出")
    print()
    
    # 获取用户输入
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        try:
            choice = input("请选择 (1/2/3): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 再见！")
            return
    
    # 处理选择
    if choice in ['1', 'openvla']:
        policy_type = 'openvla'
    elif choice in ['2', 'tinyvla']:
        policy_type = 'tinyvla'
    elif choice in ['3', 'exit', 'quit', 'q']:
        print("👋 再见！")
        return
    else:
        print(f"❌ 无效选择: {choice}")
        return
    
    # 更新配置
    print()
    print(f"正在切换到 {policy_type.upper()}...")
    
    if update_inference_server(policy_type):
        print()
        print("=" * 60)
        print("✅ 切换完成！")
        print("=" * 60)
        print()
        print("下一步:")
        print("  1. 重启 inference_server.py")
        print("  2. 确保模型路径正确")
        
        if policy_type == 'openvla':
            print("  3. OpenVLA 模型路径: ~/Desktop/openvla/openvla-7b")
            print("  4. 需要约 14GB GPU 内存")
        else:
            print("  3. TinyVLA 模型路径: /home/tianxiaoyan/TinyVLA/...")
        
        print()
    else:
        print()
        print("❌ 切换失败，请检查错误信息")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
