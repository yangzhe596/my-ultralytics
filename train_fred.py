#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# -*- coding: utf-8 -*-
"""
FRED 数据集训练脚本 - YAML配置版本
从YAML配置文件中读取所有训练参数.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from ultralytics import YOLO

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def load_config(config_path):
    """加载YAML配置文件."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        LOGGER.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        LOGGER.error(f"加载配置文件失败: {e}")
        sys.exit(1)


def get_dataset_config(config):
    """获取数据集配置."""
    dataset_config = config["dataset"]
    use_simple = dataset_config["use_simple"]

    if use_simple:
        return dataset_config["simple"]
    else:
        return dataset_config["full"]


def get_training_args(config, dataset_cfg, config_path, project_root):
    """构建训练参数."""
    training_cfg = config["training"]
    model_cfg = config["model"]

    # 基础训练参数
    train_args = {
        "data": str(config_path),
        "epochs": training_cfg["epochs"],
        "imgsz": training_cfg["imgsz"],
        "batch": training_cfg["batch_size"],
        "device": training_cfg["device"],
        "project": str(project_root / training_cfg["project"]),
        "name": f"{dataset_cfg['name_prefix']}_{model_cfg['size']}",
        "exist_ok": training_cfg["exist_ok"],
        "pretrained": training_cfg["pretrained"],
    }

    # 学习率和优化器参数
    train_args.update(
        {
            "lr0": training_cfg["lr0"],
            "lrf": training_cfg.get("lrf", training_cfg["lr0"] * 0.01),
            "optimizer": training_cfg["optimizer"],
            "momentum": training_cfg["momentum"],
            "weight_decay": training_cfg["weight_decay"],
        }
    )

    # 学习率调度参数
    train_args.update(
        {
            "warmup_epochs": training_cfg["warmup_epochs"],
            "warmup_momentum": training_cfg["warmup_momentum"],
            "warmup_bias_lr": training_cfg["warmup_bias_lr"],
        }
    )

    # 数据增强参数
    train_args.update(
        {
            "hsv_h": training_cfg["hsv_h"],
            "hsv_s": training_cfg["hsv_s"],
            "hsv_v": training_cfg["hsv_v"],
            "degrees": training_cfg["degrees"],
            "translate": training_cfg["translate"],
            "scale": training_cfg["scale"],
            "shear": training_cfg["shear"],
            "perspective": training_cfg["perspective"],
            "flipud": training_cfg["flipud"],
            "fliplr": training_cfg["fliplr"],
            "mosaic": training_cfg["mosaic"],
            "mixup": training_cfg["mixup"],
        }
    )

    # 训练控制参数
    train_args.update(
        {
            "patience": training_cfg["patience"],
            "save_period": training_cfg["save_period"],
            "workers": training_cfg["workers"],
            "close_mosaic": training_cfg["close_mosaic"],
        }
    )

    # 输出参数
    output_cfg = config["output"]
    train_args.update(
        {
            "conf": output_cfg["conf"],
            "iou": output_cfg["iou"],
            "max_det": output_cfg["max_det"],
        }
    )

    # 可视化参数
    viz_cfg = config.get("visualization", {})
    train_args.update(
        {
            "plots": viz_cfg.get("plots", False),
            "visualize": viz_cfg.get("visualize", False),
        }
    )

    # 高级参数
    adv_cfg = config.get("advanced", {})
    train_args.update(
        {
            "rect": adv_cfg.get("rect", False),
            "cos_lr": adv_cfg.get("cos_lr", False),
            "nbs": adv_cfg.get("nbs", 64),
            "amp": False,  # 禁用自动混合精度以防止NaN
        }
    )

    # 冻结参数
    train_args["freeze"] = dataset_cfg.get("freeze", 0)

    return train_args, model_cfg


def train_model(config_path, config):
    """训练模型."""
    # 获取数据集配置
    dataset_cfg = get_dataset_config(config)
    LOGGER.info(f"使用数据集配置: {dataset_cfg['name_prefix']}")

    # 获取模型配置
    model_cfg = config["model"]
    model_name = f"yolov8{model_cfg['size']}.pt"
    LOGGER.info(f"加载预训练模型: {model_name}")

    # 加载模型
    try:
        model = YOLO(model_name)
    except Exception as e:
        LOGGER.error(f"加载模型失败: {e}")
        return False

    # 获取项目根目录
    project_root = Path(__file__).parent

    # 构建训练参数
    train_args, model_cfg = get_training_args(config, dataset_cfg, config_path, project_root)

    # 获取冻结配置
    freeze_layers = dataset_cfg.get("freeze", 0)
    freeze_epochs = dataset_cfg.get("freeze_epochs", 0)
    test_after_train = config["training"]["test_after_train"]

    # 如果需要冻结训练，进行两阶段训练
    if freeze_layers > 0 and freeze_epochs > 0:
        LOGGER.info(
            f"使用两阶段训练：冻结 {freeze_layers} 层训练 {freeze_epochs} 轮，然后解冻全量训练 {config['training']['epochs'] - freeze_epochs} 轮"
        )

        # 第一阶段：冻结训练
        LOGGER.info("=== 第一阶段：冻结训练 ===")
        freeze_args = train_args.copy()
        freeze_args.update(
            {
                "epochs": freeze_epochs,
                "name": f"{dataset_cfg['name_prefix']}_{model_cfg['size']}_freeze",
                "freeze": freeze_layers,
                "lr0": model_cfg["stage1_lr"],
            }
        )

        try:
            results = model.train(**freeze_args)
            LOGGER.info("第一阶段训练完成!")

            # 使用第一阶段训练的结果继续训练
            model = YOLO(str(results.save_dir / "weights/best.pt"))

        except Exception as e:
            LOGGER.error(f"第一阶段训练过程中出现错误: {e}")
            return False

        # 第二阶段：解冻训练
        LOGGER.info("=== 第二阶段：解冻全量训练 ===")
        unfreeze_args = train_args.copy()
        unfreeze_args.update(
            {
                "name": f"{dataset_cfg['name_prefix']}_{model_cfg['size']}_full",
                "freeze": 0,  # 解冻所有层
                "lr0": model_cfg["stage2_lr"],  # 降低学习率
                "pretrained": False,  # 使用第一阶段训练的模型
                # 移除 'resume': True，改为使用第一阶段训练的模型作为起点
            }
        )

        train_args = unfreeze_args
    else:
        # 单阶段训练
        train_args["freeze"] = freeze_layers

    # 开始训练
    LOGGER.info("开始训练...")
    try:
        results = model.train(**train_args)
        LOGGER.info("训练完成!")

        # 保存最佳模型路径
        best_model_path = results.save_dir / "weights/best.pt"
        LOGGER.info(f"最佳模型保存在: {best_model_path}")

        # 运行验证
        LOGGER.info("运行验证...")
        metrics = model.val(data=str(config_path))
        LOGGER.info(f"验证结果: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

        # 训练完成后在测试集上评估
        if test_after_train:
            LOGGER.info("\n=== 在测试集上评估最佳模型 ===")
            try:
                # 在测试集上评估模型
                test_results = model.val(
                    data=str(config_path),
                    split="test",  # 指定使用测试集
                    imgsz=config["training"]["imgsz"],
                    device=config["training"]["device"],
                    batch=config["training"]["batch_size"],
                    project=str(project_root / "runs/test"),
                    name=f"{dataset_cfg['name_prefix']}_{model_cfg['size']}_test",
                    exist_ok=True,
                )

                # 打印测试结果
                LOGGER.info("\n测试集评估结果:")
                LOGGER.info(f"  mAP50: {test_results.box.map50:.4f}")
                LOGGER.info(f"  mAP50-95: {test_results.box.map:.4f}")
                LOGGER.info(f"  精确率: {test_results.box.mp:.4f}")
                LOGGER.info(f"  召回率: {test_results.box.mr:.4f}")

                # 保存测试结果
                test_results_path = results.save_dir / "test_results.txt"
                with open(test_results_path, "w") as f:
                    f.write(f"mAP50: {test_results.box.map50:.4f}\n")
                    f.write(f"mAP50-95: {test_results.box.map:.4f}\n")
                    f.write(f"Precision: {test_results.box.mp:.4f}\n")
                    f.write(f"Recall: {test_results.box.mr:.4f}\n")

                LOGGER.info(f"测试结果已保存到: {test_results_path}")

            except Exception as e:
                LOGGER.warning(f"测试集评估失败: {e}")
                LOGGER.info("这可能是因为数据集中没有测试集，或者测试集为空")

        return True

    except Exception as e:
        LOGGER.error(f"训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="FRED 数据集训练脚本 - YAML配置版本")
    parser.add_argument("--config", type=str, default="datasets/fred_train_config.yaml", help="训练配置文件路径")
    parser.add_argument("--dataset-config", type=str, help="数据集配置文件路径（覆盖配置文件中的设置）")

    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)

    # 如果指定了数据集配置文件，覆盖配置文件中的设置
    if args.dataset_config:
        config["dataset"]["full"]["config_file"] = args.dataset_config
        config["dataset"]["simple"]["config_file"] = args.dataset_config

    # 获取数据集配置文件路径
    dataset_cfg = get_dataset_config(config)
    config_path = dataset_cfg["config_file"]

    # 检查数据集配置文件是否存在
    if not Path(config_path).exists():
        LOGGER.error(f"数据集配置文件不存在: {config_path}")
        return 1

    # 训练模型
    success = train_model(config_path, config)

    if success:
        LOGGER.info("\n✅ 训练成功完成!")
        return 0
    else:
        LOGGER.error("\n❌ 训练失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
