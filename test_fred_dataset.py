#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""测试fred_simple数据集是否能正常加载."""

import sys
from pathlib import Path

# 添加ultralytics到路径
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO


def test_dataset():
    """测试数据集加载."""
    # 设置路径
    project_root = Path(__file__).parent
    config_path = project_root / "datasets/fred_simple.yaml"

    print("=== 测试fred_simple数据集 ===")
    print(f"项目根目录: {project_root}")
    print(f"配置文件: {config_path}")
    print()

    # 检查配置文件是否存在
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return False

    # 加载模型
    try:
        print("加载YOLOv8n模型...")
        model = YOLO("yolov8n.pt")
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

    # 测试数据集加载
    try:
        print("测试数据集加载...")
        model.info()

        # 验证数据集
        print("验证数据集配置...")
        metrics = model.val(data=str(config_path), epochs=1, imgsz=640, batch=8)

        print("数据集测试成功!")
        print(f"验证结果: mAP50={metrics.box.map50:.4f}")
        return True

    except Exception as e:
        print(f"数据集测试失败: {e}")
        return False


if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n✅ 数据集测试通过，可以开始训练")
    else:
        print("\n❌ 数据集测试失败，请检查配置")
