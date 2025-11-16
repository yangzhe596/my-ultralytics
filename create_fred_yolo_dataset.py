#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# -*- coding: utf-8 -*-
"""
FRED 数据集转换为 YOLO 格式.

基于 FRED 官方数据集生成指南，生成与 Ultralytics YOLO 完全兼容的数据集格式。
标签计算方法与 convert_fred_to_coco_v2.py 保持一致，确保数据一致性。

主要功能：
1. 使用 interpolated_coordinates.txt（包含 drone_id，支持多目标追踪）
2. 支持帧级别划分（Frame-level Split）- 数据分布更均衡
3. 支持序列级别划分（Sequence-level Split）- 更好的泛化评估
4. 生成 YOLO 格式的标签文件（.txt）
5. 创建 YOLO 数据集配置文件（.yaml）
6. 完整的数据验证和统计

使用方法：
    # 帧级别划分（推荐，使用符号链接）
    python create_fred_yolo_dataset.py --split-mode frame --modality both

    # 序列级别划分（使用符号链接）
    python create_fred_yolo_dataset.py --split-mode sequence --modality both

    # 仅转换 RGB 模态
    python create_fred_yolo_dataset.py --modality rgb

    # 生成简化数据集（训练/验证/测试各100张）
    python create_fred_yolo_dataset.py --simple-dataset --simple-samples 100

    # 复制文件而非使用符号链接
    python create_fred_yolo_dataset.py --copy-files

    # 禁用符号链接（与 --copy-files 相同）
    python create_fred_yolo_dataset.py --no-use-symlinks
"""

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FREDtoYOLOConverter:
    """FRED 数据集转换为 YOLO 格式."""

    def __init__(self, fred_root, output_root, split_mode="frame"):
        """初始化转换器.

        Args:
            fred_root: FRED 数据集根目录
            output_root: 输出目录
            split_mode: 'frame' 或 'sequence'
        """
        self.fred_root = Path(fred_root)
        self.output_root = Path(output_root)
        self.split_mode = split_mode

        if not self.fred_root.exists():
            raise FileNotFoundError(f"FRED 数据集根目录不存在: {self.fred_root}")

        # YOLO 类别定义（与 convert_fred_to_coco_v2.py 保持一致）
        self.class_names = ["drone"]
        self.num_classes = len(self.class_names)

        logger.info(f"FRED 根目录: {self.fred_root}")
        logger.info(f"输出目录: {self.output_root}")
        logger.info(f"划分模式: {split_mode}")

    def get_all_sequences(self):
        """获取所有可用的序列 ID."""
        sequences = []
        for seq_dir in sorted(self.fred_root.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                sequences.append(int(seq_dir.name))
        return sorted(sequences)

    def load_annotations(self, sequence_path):
        """加载标注文件（与 convert_fred_to_coco_v2.py 保持一致）.

        Args:
            sequence_path: 序列目录路径

        Returns:
            dict: {timestamp: [{'bbox': (x1,y1,x2,y2), 'drone_id': id}, ...]}
        """
        # 优先使用插值标注文件
        annotation_file = sequence_path / "interpolated_coordinates.txt"

        if not annotation_file.exists():
            logger.warning("未找到 interpolated_coordinates.txt，尝试使用 coordinates.txt")
            annotation_file = sequence_path / "coordinates.txt"

        if not annotation_file.exists():
            logger.warning(f"序列 {sequence_path.name} 无标注文件")
            return {}

        annotations = {}

        with open(annotation_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析格式: "时间: x1, y1, x2, y2, id" 或 "时间: x1, y1, x2, y2"
                    time_str, coords_str = line.split(": ")
                    timestamp = float(time_str)

                    coords = [x.strip() for x in coords_str.split(",")]

                    x1, y1, x2, y2 = map(float, coords[:4])
                    drone_id = int(float(coords[4])) if len(coords) > 4 else 1

                    # 验证边界框有效性
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"{annotation_file.name} 第 {line_num} 行: 无效边界框 ({x1},{y1},{x2},{y2})")
                        continue

                    if timestamp not in annotations:
                        annotations[timestamp] = []

                    annotations[timestamp].append({"bbox": (x1, y1, x2, y2), "drone_id": drone_id})

                except Exception as e:
                    logger.warning(f"{annotation_file.name} 第 {line_num} 行解析失败: {e}")
                    continue

        logger.info(f"序列 {sequence_path.name}: 加载 {len(annotations)} 个时间戳的标注")
        return annotations

    def get_frames(self, sequence_path, modality):
        """获取帧列表及其时间戳（与 convert_fred_to_coco_v2.py 保持一致）.

        Args:
            sequence_path: 序列目录路径
            modality: 'rgb' 或 'event'

        Returns:
            list: [(timestamp, frame_path), ...]
        """
        if modality == "rgb":
            # 使用 PADDED_RGB（与 Event 对齐）
            frame_dir = sequence_path / "PADDED_RGB"
            if not frame_dir.exists():
                # 回退到原始 RGB
                frame_dir = sequence_path / "RGB"
            pattern = "*.jpg"
        elif modality == "event":
            frame_dir = sequence_path / "Event" / "Frames"
            pattern = "*.png"
        else:
            raise ValueError(f"未知模态: {modality}")

        if not frame_dir.exists():
            logger.warning(f"帧目录不存在: {frame_dir}")
            return []

        frames = []

        for frame_path in sorted(frame_dir.glob(pattern)):
            timestamp = self._extract_timestamp(frame_path.name, modality)
            if timestamp is not None:
                frames.append((timestamp, frame_path))

        # 按时间戳排序
        frames = sorted(frames, key=lambda x: x[0])

        # 转换为相对时间戳
        if frames:
            first_timestamp = frames[0][0]
            frames = [(t - first_timestamp, path) for t, path in frames]

        return frames

    def _extract_timestamp(self, filename, modality):
        """从文件名提取时间戳（与 convert_fred_to_coco_v2.py 保持一致）."""
        try:
            if modality == "rgb":
                # Video_0_16_03_03.363444.jpg
                name = filename.replace(".jpg", "")
                parts = name.split("_")

                if len(parts) >= 4:
                    time_parts = parts[-3:]
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = float(time_parts[2])

                    return hours * 3600 + minutes * 60 + seconds

            elif modality == "event":
                # Video_0_frame_100032333.png
                name = filename.replace(".png", "")
                parts = name.split("_")

                if len(parts) >= 3:
                    timestamp_us = int(parts[-1])
                    return timestamp_us / 1_000_000

        except Exception as e:
            logger.warning(f"无法从文件名 '{filename}' 提取时间戳: {e}")

        return None

    def find_closest_annotation(self, timestamp, annotations, threshold=0.05):
        """查找最接近的标注（与 convert_fred_to_coco_v2.py 保持一致）.

        Args:
            timestamp: 目标时间戳
            annotations: 标注字典
            threshold: 时间容差（秒）

        Returns:
            list: 标注列表或空列表
        """
        if not annotations:
            return []

        closest_time = min(annotations.keys(), key=lambda t: abs(t - timestamp))

        if abs(closest_time - timestamp) <= threshold:
            return annotations[closest_time]

        return []

    def validate_bbox(self, bbox, width, height):
        """验证并修正边界框（与 convert_fred_to_coco_v2.py 保持一致）.

        Args:
            bbox: (x1, y1, x2, y2)
            width: 图像宽度
            height: 图像高度

        Returns:
            tuple: (is_valid, corrected_bbox)
        """
        x1, y1, x2, y2 = bbox

        # 确保坐标顺序正确
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # 限制在图像边界内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # 确保有效面积
        if x2 <= x1 or y2 <= y1:
            return False, None

        return True, (x1, y1, x2, y2)

    def convert_bbox_to_yolo(self, bbox, width, height):
        """将边界框转换为 YOLO 格式.

        Args:
            bbox: (x1, y1, x2, y2) 边界框坐标
            width: 图像宽度
            height: 图像高度

        Returns:
            tuple: (class_id, x_center, y_center, bbox_width, bbox_height) 归一化坐标
        """
        x1, y1, x2, y2 = bbox

        # 计算中心点和宽高
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # 归一化
        x_center /= width
        y_center /= height
        bbox_width /= width
        bbox_height /= height

        # 确保在 [0, 1] 范围内
        x_center = max(0, min(x_center, 1.0))
        y_center = max(0, min(y_center, 1.0))
        bbox_width = max(0, min(bbox_width, 1.0))
        bbox_height = max(0, min(bbox_height, 1.0))

        # 类别 ID（drone = 0）
        class_id = 0

        return class_id, x_center, y_center, bbox_width, bbox_height

    def process_sequence(self, sequence_id, modality, output_dir, split, use_symlinks=True):
        """处理单个序列，生成 YOLO 格式的图像和标签.

        Args:
            sequence_id: 序列 ID
            modality: 'rgb' 或 'event'
            output_dir: 输出目录
            split: 'train', 'val', 或 'test'
            use_symlinks: 是否使用符号链接/相对路径而非复制文件

        Returns:
            dict: 统计信息
        """
        sequence_path = self.fred_root / str(sequence_id)

        if not sequence_path.exists():
            logger.warning(f"序列 {sequence_id} 不存在")
            return {}

        # 创建输出目录
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 加载标注
        annotations_dict = self.load_annotations(sequence_path)

        # 获取帧
        frames = self.get_frames(sequence_path, modality)

        if not frames:
            logger.warning(f"序列 {sequence_id} ({modality}) 无帧")
            return {}

        # 统计信息
        stats = {
            "total_frames": len(frames),
            "matched_frames": 0,
            "total_annotations": 0,
            "invalid_bboxes": 0,
            "processed_images": 0,
        }

        # 图像尺寸（FRED 数据集固定尺寸）
        width, height = 1280, 720

        # 创建图像列表文件（记录图像路径）
        image_list_file = images_dir / f"{split}_images.txt"

        for timestamp, frame_path in tqdm(frames, desc=f"处理序列 {sequence_id} ({modality})"):
            # 查找匹配的标注
            anns = self.find_closest_annotation(timestamp, annotations_dict)

            # 仅包含有标注的帧
            if not anns:
                continue

            stats["matched_frames"] += 1

            # 处理图像路径
            image_ext = frame_path.suffix
            output_image_name = f"{sequence_id:04d}_{timestamp:.6f}{image_ext}"

            # 使用绝对路径创建符号链接
            absolute_path = str(frame_path.resolve())

            if use_symlinks:
                # 创建符号链接到绝对路径
                output_image_path = images_dir / output_image_name

                try:
                    # 创建绝对路径的符号链接
                    os.symlink(absolute_path, output_image_path)
                    stats["processed_images"] += 1
                except (OSError, NotImplementedError) as e:
                    # 如果符号链接失败，创建文本文件记录路径
                    logger.warning(f"创建符号链接失败，使用.path文件: {e}")
                    link_file = images_dir / f"{output_image_name}.path"
                    with open(link_file, "w") as f:
                        f.write(absolute_path)
                    stats["processed_images"] += 1
            else:
                # 复制图像文件（原始方法）
                output_image_path = images_dir / output_image_name
                try:
                    shutil.copy2(frame_path, output_image_path)
                    stats["processed_images"] += 1
                except Exception as e:
                    logger.warning(f"复制图像失败 {frame_path}: {e}")
                    continue

            # 将图像路径添加到列表文件（使用相对于数据集根目录的路径）
            try:
                list_path = os.path.relpath(frame_path, output_dir.parent)
            except ValueError:
                list_path = str(frame_path)

            with open(image_list_file, "a") as f:
                f.write(f"{list_path}\n")

            # 创建对应的标签文件（使用图像文件的基础名称）
            output_label_path = labels_dir / f"{Path(output_image_name).stem}.txt"

            with open(output_label_path, "w") as f:
                for ann in anns:
                    bbox = ann["bbox"]

                    # 验证边界框
                    is_valid, corrected_bbox = self.validate_bbox(bbox, width, height)

                    if not is_valid:
                        stats["invalid_bboxes"] += 1
                        continue

                    # 转换为 YOLO 格式
                    class_id, x_center, y_center, bbox_width, bbox_height = self.convert_bbox_to_yolo(
                        corrected_bbox, width, height
                    )

                    # 写入 YOLO 格式的标签
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    stats["total_annotations"] += 1

        return stats

    def split_sequences(self, sequences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """序列级别划分（与 convert_fred_to_coco_v2.py 保持一致）.

        Args:
            sequences: 序列 ID 列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子

        Returns:
            tuple: (train_seqs, val_seqs, test_seqs)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        random.seed(seed)
        sequences = sequences.copy()
        random.shuffle(sequences)

        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train : n_train + n_val]
        test_seqs = sequences[n_train + n_val :]

        return sorted(train_seqs), sorted(val_seqs), sorted(test_seqs)

    def get_frame_split(self, sequence_id, frame_idx, train_ratio=0.7, val_ratio=0.15, seed=42):
        """帧级别划分（与 convert_fred_to_coco_v2.py 保持一致）.

        Args:
            sequence_id: 序列 ID
            frame_idx: 帧索引
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子

        Returns:
            str: 'train', 'val', 或 'test'
        """
        hash_input = f"{sequence_id}_{frame_idx}_{seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        rand_val = (hash_value % 1000000) / 1000000.0

        if rand_val < train_ratio:
            return "train"
        elif rand_val < train_ratio + val_ratio:
            return "val"
        else:
            return "test"

    def create_dataset_yaml(self, output_dir, modality):
        """创建 YOLO 数据集配置文件.

        Args:
            output_dir: 输出目录
            modality: 'rgb' 或 'event'
        """
        dataset_name = f"fred_{modality}"

        yaml_content = f"""# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# FRED {modality.upper()} dataset converted from FRED format
# Generated by create_fred_yolo_dataset.py
# Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Train/val/test sets
path: {dataset_name}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
names:
"""

        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"

        yaml_file = output_dir / f"{dataset_name}.yaml"
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        logger.info(f"✓ 数据集配置文件已创建: {yaml_file}")

    def generate_simple_dataset(self, modality="rgb", num_samples=100, seed=42, use_symlinks=True):
        """生成简化的 YOLO 数据集，用于快速验证.

        Args:
            modality: 'rgb' 或 'event'
            num_samples: 每个划分的样本数量
            seed: 随机种子
            use_symlinks: 是否使用符号链接/相对路径而非复制文件
        """
        logger.info(f"\n{'=' * 70}")
        logger.info("FRED 转 YOLO - 简化数据集生成")
        logger.info(f"{'=' * 70}")
        logger.info(f"模态: {modality}")
        logger.info(f"每个划分样本数: {num_samples}")

        # 创建输出目录
        dataset_dir = self.output_root / f"fred_{modality}_simple"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 创建图像和标签目录
        for split in ["train", "val", "test"]:
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # 获取所有序列
        sequences = self.get_all_sequences()

        # 收集所有帧
        all_frames = []
        for seq_id in sequences:
            sequence_path = self.fred_root / str(seq_id)
            frames = self.get_frames(sequence_path, modality)

            for frame_idx, (timestamp, frame_path) in enumerate(frames):
                all_frames.append((seq_id, timestamp, frame_path))

        # 随机打乱并选择样本
        random.seed(seed)
        random.shuffle(all_frames)

        # 选择总样本数（每个划分num_samples，共3*num_samples）
        total_samples = num_samples * 3
        selected_frames = all_frames[: min(total_samples, len(all_frames))]

        # 平均分配到三个划分
        train_frames = selected_frames[:num_samples]
        val_frames = selected_frames[num_samples : num_samples * 2]
        test_frames = (
            selected_frames[num_samples * 2 : num_samples * 3]
            if len(selected_frames) >= num_samples * 3
            else selected_frames[num_samples * 2 :]
        )

        # 处理每个划分
        splits = [("train", train_frames), ("val", val_frames), ("test", test_frames)]

        total_stats = {"total_frames": 0, "processed_frames": 0, "total_annotations": 0, "invalid_bboxes": 0}

        for split_name, frames in splits:
            logger.info(f"\n处理 {split_name} 划分: {len(frames)} 帧")

            # 创建图像列表文件（记录图像路径）
            images_dir = dataset_dir / "images" / split_name
            labels_dir = dataset_dir / "labels" / split_name
            image_list_file = images_dir / f"{split_name}_images.txt"

            # 按序列分组处理
            seq_frames = defaultdict(list)
            for seq_id, timestamp, frame_path in frames:
                seq_frames[seq_id].append((timestamp, frame_path))

            for seq_id, seq_frame_list in seq_frames.items():
                sequence_path = self.fred_root / str(seq_id)
                annotations_dict = self.load_annotations(sequence_path)

                for timestamp, frame_path in tqdm(seq_frame_list, desc=f"处理序列 {seq_id} ({split_name})"):
                    # 查找匹配的标注
                    anns = self.find_closest_annotation(timestamp, annotations_dict)

                    # 仅包含有标注的帧
                    if not anns:
                        continue

                    total_stats["total_frames"] += 1

                    # 处理图像路径
                    image_ext = frame_path.suffix
                    output_image_name = f"{seq_id:04d}_{timestamp:.6f}{image_ext}"

                    # 使用绝对路径创建符号链接
                    absolute_path = str(frame_path.resolve())

                    # 创建符号链接到绝对路径
                    output_image_path = images_dir / output_image_name
                    try:
                        # 创建绝对路径的符号链接
                        os.symlink(absolute_path, output_image_path)
                        total_stats["processed_frames"] += 1
                    except (OSError, NotImplementedError) as e:
                        # 如果符号链接失败，创建文本文件记录路径
                        logger.warning(f"创建符号链接失败，使用.path文件: {e}")
                        link_file = images_dir / f"{output_image_name}.path"
                        with open(link_file, "w") as f:
                            f.write(absolute_path)
                        total_stats["processed_frames"] += 1

                    # 将图像路径添加到列表文件（使用相对于数据集根目录的路径）
                    try:
                        list_path = os.path.relpath(frame_path, dataset_dir.parent)
                    except ValueError:
                        list_path = str(frame_path)

                    with open(image_list_file, "a") as f:
                        f.write(f"{list_path}\n")

                    # 创建标签文件（使用图像文件的基础名称）
                    output_label_path = labels_dir / f"{Path(output_image_name).stem}.txt"

                    width, height = 1280, 720

                    with open(output_label_path, "w") as f:
                        for ann in anns:
                            bbox = ann["bbox"]

                            # 验证边界框
                            is_valid, corrected_bbox = self.validate_bbox(bbox, width, height)

                            if not is_valid:
                                total_stats["invalid_bboxes"] += 1
                                continue

                            # 转换为 YOLO 格式
                            class_id, x_center, y_center, bbox_width, bbox_height = self.convert_bbox_to_yolo(
                                corrected_bbox, width, height
                            )

                            # 写入 YOLO 格式的标签
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                            total_stats["total_annotations"] += 1

            logger.info(f"\n{split_name} 统计:")
            logger.info(f"  处理图像: {len(frames)}")

        # 创建数据集配置文件
        dataset_name = f"fred_{modality}_simple"
        yaml_content = f"""# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# FRED {modality.upper()} Simple Dataset for Quick Validation
# Generated by create_fred_yolo_dataset.py
# Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Samples per split: {num_samples}

# Train/val/test sets
path: {dataset_name}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
names:
"""

        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"

        yaml_file = self.output_root / f"{dataset_name}.yaml"
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        logger.info(f"✓ 数据集配置文件已创建: {yaml_file}")

        # 保存划分信息
        split_info = {
            "dataset_type": "simple",
            "samples_per_split": num_samples,
            "total_samples": total_stats["processed_frames"],
            "seed": seed,
            "modality": modality,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, "w") as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"\n{'=' * 70}")
        logger.info("简化数据集生成完成!")
        logger.info(f"{'=' * 70}")
        logger.info(f"总帧数: {total_stats['total_frames']}")
        logger.info(f"处理帧数: {total_stats['processed_frames']}")
        logger.info(f"总标注: {total_stats['total_annotations']}")
        logger.info(f"无效框: {total_stats['invalid_bboxes']}")

    def generate_all_splits(
        self, modality="both", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, use_symlinks=True
    ):
        """生成所有划分和模态的 YOLO 数据集.

        Args:
            modality: 'rgb', 'event', 或 'both'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            use_symlinks: 是否使用符号链接/相对路径而非复制文件
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"FRED 转 YOLO - {self.split_mode.upper()} 级别划分")
        logger.info(f"{'=' * 70}")

        modalities = ["rgb", "event"] if modality == "both" else [modality]

        for mod in modalities:
            logger.info(f"\n处理 {mod.upper()} 模态...")

            # 创建输出目录
            dataset_dir = self.output_root / f"fred_{mod}"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # 创建图像和标签目录
            for split in ["train", "val", "test"]:
                (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

            # 获取所有序列
            sequences = self.get_all_sequences()

            if self.split_mode == "sequence":
                # 序列级别划分
                train_seqs, val_seqs, test_seqs = self.split_sequences(
                    sequences, train_ratio, val_ratio, test_ratio, seed
                )

                logger.info("\n序列划分:")
                logger.info(f"  训练: {len(train_seqs)} 序列")
                logger.info(f"  验证: {len(val_seqs)} 序列")
                logger.info(f"  测试: {len(test_seqs)} 序列")

                # 处理每个划分
                for split_name, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
                    total_stats = defaultdict(int)

                    for seq_id in tqdm(seqs, desc=f"处理 {split_name} 序列"):
                        stats = self.process_sequence(seq_id, mod, dataset_dir, split_name, use_symlinks=True)

                        for key, value in stats.items():
                            total_stats[key] += value

                    logger.info(f"\n{split_name} 统计:")
                    logger.info(f"  处理图像: {total_stats['processed_images']}")
                    logger.info(f"  匹配帧: {total_stats['matched_frames']}")
                    logger.info(f"  总标注: {total_stats['total_annotations']}")
                    logger.info(f"  无效框: {total_stats['invalid_bboxes']}")

            else:  # frame-level
                logger.info("\n帧级别划分...")

                # 收集所有帧并划分
                all_frames = []
                for seq_id in sequences:
                    sequence_path = self.fred_root / str(seq_id)
                    frames = self.get_frames(sequence_path, mod)

                    for frame_idx, (timestamp, frame_path) in enumerate(frames):
                        split = self.get_frame_split(seq_id, frame_idx, train_ratio, val_ratio, seed)
                        all_frames.append((seq_id, timestamp, frame_path, split))

                # 按划分分组
                split_frames = defaultdict(list)
                for seq_id, timestamp, frame_path, split in all_frames:
                    split_frames[split].append((seq_id, timestamp, frame_path))

                # 处理每个划分
                total_stats = defaultdict(lambda: defaultdict(int))

                for split_name, frames in split_frames.items():
                    logger.info(f"\n处理 {split_name} 划分: {len(frames)} 帧")

                    # 创建图像列表文件（记录图像路径）
                    images_dir = dataset_dir / "images" / split_name
                    labels_dir = dataset_dir / "labels" / split_name
                    image_list_file = images_dir / f"{split_name}_images.txt"

                    # 按序列分组处理
                    seq_frames = defaultdict(list)
                    for seq_id, timestamp, frame_path in frames:
                        seq_frames[seq_id].append((timestamp, frame_path))

                    for seq_id, seq_frame_list in seq_frames.items():
                        sequence_path = self.fred_root / str(seq_id)
                        annotations_dict = self.load_annotations(sequence_path)

                        for timestamp, frame_path in tqdm(seq_frame_list, desc=f"处理序列 {seq_id} ({split_name})"):
                            # 查找匹配的标注
                            anns = self.find_closest_annotation(timestamp, annotations_dict)

                            # 仅包含有标注的帧
                            if not anns:
                                continue

                            total_stats[split_name]["matched_frames"] += 1

                            # 处理图像路径
                            image_ext = frame_path.suffix
                            output_image_name = f"{seq_id:04d}_{timestamp:.6f}{image_ext}"

                            # 使用绝对路径创建符号链接
                            absolute_path = str(frame_path.resolve())

                            # 创建符号链接到绝对路径
                            output_image_path = images_dir / output_image_name
                            try:
                                # 创建绝对路径的符号链接
                                os.symlink(absolute_path, output_image_path)
                                total_stats[split_name]["processed_images"] += 1
                            except (OSError, NotImplementedError) as e:
                                # 如果符号链接失败，创建文本文件记录路径
                                logger.warning(f"创建符号链接失败，使用.path文件: {e}")
                                link_file = images_dir / f"{output_image_name}.path"
                                with open(link_file, "w") as f:
                                    f.write(absolute_path)
                                total_stats[split_name]["processed_images"] += 1

                            # 将图像路径添加到列表文件（使用相对于数据集根目录的路径）
                            try:
                                list_path = os.path.relpath(frame_path, dataset_dir.parent)
                            except ValueError:
                                list_path = str(frame_path)

                            with open(image_list_file, "a") as f:
                                f.write(f"{list_path}\n")

                            # 创建标签文件（使用图像文件的基础名称）
                            output_label_path = labels_dir / f"{Path(output_image_name).stem}.txt"

                            width, height = 1280, 720

                            with open(output_label_path, "w") as f:
                                for ann in anns:
                                    bbox = ann["bbox"]

                                    # 验证边界框
                                    is_valid, corrected_bbox = self.validate_bbox(bbox, width, height)

                                    if not is_valid:
                                        total_stats[split_name]["invalid_bboxes"] += 1
                                        continue

                                    # 转换为 YOLO 格式
                                    class_id, x_center, y_center, bbox_width, bbox_height = self.convert_bbox_to_yolo(
                                        corrected_bbox, width, height
                                    )

                                    # 写入 YOLO 格式的标签
                                    f.write(
                                        f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                                    )
                                    total_stats[split_name]["total_annotations"] += 1

                    logger.info(f"\n{split_name} 统计:")
                    logger.info(f"  处理图像: {total_stats[split_name]['processed_images']}")
                    logger.info(f"  匹配帧: {total_stats[split_name]['matched_frames']}")
                    logger.info(f"  总标注: {total_stats[split_name]['total_annotations']}")
                    logger.info(f"  无效框: {total_stats[split_name]['invalid_bboxes']}")

            # 创建数据集配置文件
            self.create_dataset_yaml(self.output_root, mod)

        # 保存划分信息
        split_info = {
            "split_mode": self.split_mode,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "modalities": modalities,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        info_file = self.output_root / "split_info.json"
        with open(info_file, "w") as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"\n✓ 划分信息已保存: {info_file}")

        # 验证符号链接（如果使用了符号链接）
        if use_symlinks:
            logger.info(f"\n{'=' * 70}")
            logger.info("验证符号链接...")
            logger.info(f"{'=' * 70}")

            for mod in modalities:
                validate_symlinks(self.output_root, mod)

        logger.info(f"\n{'=' * 70}")
        logger.info("转换完成！")
        logger.info(f"{'=' * 70}")


def validate_symlinks(dataset_dir, modality):
    """验证符号链接是否有效.

    Args:
        dataset_dir: 数据集目录
        modality: 'rgb' 或 'event'

    Returns:
        dict: 验证结果统计
    """
    dataset_name = f"fred_{modality}"
    data_path = dataset_dir / dataset_name

    if not data_path.exists():
        logger.warning(f"数据集目录不存在: {data_path}")
        return {}

    stats = {"total_links": 0, "valid_links": 0, "broken_links": 0, "path_files": 0}

    for split in ["train", "val", "test"]:
        images_dir = data_path / "images" / split
        if not images_dir.exists():
            continue

        logger.info(f"\n验证 {modality} {split} 集符号链接...")

        for item in images_dir.iterdir():
            if item.is_symlink():
                stats["total_links"] += 1

                # 检查符号链接是否有效
                try:
                    if item.resolve().exists():
                        stats["valid_links"] += 1
                    else:
                        stats["broken_links"] += 1
                        logger.warning(f"失效链接: {item} -> {item.readlink()}")
                except Exception as e:
                    stats["broken_links"] += 1
                    logger.warning(f"链接错误 {item}: {e}")

            elif item.suffix == ".path":
                # 处理.path文件
                stats["path_files"] += 1
                try:
                    with open(item) as f:
                        path = f.read().strip()
                        if Path(path).exists():
                            stats["valid_links"] += 1
                        else:
                            stats["broken_links"] += 1
                            logger.warning(f".path文件中路径不存在: {path}")
                except Exception as e:
                    stats["broken_links"] += 1
                    logger.warning(f"读取.path文件失败 {item}: {e}")

    logger.info(f"\n{modality} 模态符号链接验证结果:")
    logger.info(f"  总链接数: {stats['total_links']}")
    logger.info(f"  有效链接: {stats['valid_links']}")
    logger.info(f"  失效链接: {stats['broken_links']}")
    logger.info(f"  路径文件: {stats['path_files']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="FRED 数据集转换为 YOLO 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 帧级别划分（推荐，使用符号链接）
  python create_fred_yolo_dataset.py --split-mode frame --modality both
  
  # 序列级别划分（使用符号链接）
  python create_fred_yolo_dataset.py --split-mode sequence --modality both
  
  # 仅转换 RGB 模态
  python create_fred_yolo_dataset.py --modality rgb
  
  # 生成简化数据集（训练/验证/测试各100张）
  python create_fred_yolo_dataset.py --simple-dataset --simple-samples 100
  
  # 复制文件而非使用符号链接
  python create_fred_yolo_dataset.py --copy-files
  
  # 仅验证现有符号链接
  python create_fred_yolo_dataset.py --validate-only --modality both
  
  # 自定义划分比例
  python create_fred_yolo_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """,
    )

    parser.add_argument("--fred-root", type=str, default="/mnt/data/datasets/fred", help="FRED 数据集根目录")
    parser.add_argument("--output-root", type=str, default="datasets/fred_yolo", help="输出目录")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="frame",
        choices=["frame", "sequence"],
        help="划分模式: frame（帧级别）或 sequence（序列级别）",
    )
    parser.add_argument("--modality", type=str, default="both", choices=["rgb", "event", "both"], help="转换的模态")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--use-symlinks", action="store_true", default=True, help="使用符号链接/相对路径而非复制文件（默认启用）"
    )
    parser.add_argument(
        "--copy-files", action="store_true", help="复制文件而非使用符号链接（覆盖 --use-symlinks 选项）"
    )
    parser.add_argument("--validate-only", action="store_true", help="仅验证现有符号链接，不执行转换")
    parser.add_argument("--simple-dataset", action="store_true", help="生成简化数据集（训练/验证/测试各指定数量）")
    parser.add_argument("--simple-samples", type=int, default=100, help="简化数据集每个划分的样本数量")

    args = parser.parse_args()

    # 如果只是验证，执行验证并退出
    if args.validate_only:
        modalities = ["rgb", "event"] if args.modality == "both" else [args.modality]
        for mod in modalities:
            validate_symlinks(Path(args.output_root), mod)
        return 0

    # 确定是否使用符号链接
    use_symlinks = args.use_symlinks and not args.copy_files

    # 如果是生成简化数据集
    if args.simple_dataset:
        modalities = [args.modality] if args.modality != "both" else ["rgb"]  # 简化数据集只生成RGB模态

        for mod in modalities:
            logger.info(f"\n生成 {mod.upper()} 简化数据集...")
            converter = FREDtoYOLOConverter(
                fred_root=args.fred_root,
                output_root=args.output_root,
                split_mode="frame",  # 简化数据集使用帧级别划分
            )

            converter.generate_simple_dataset(
                modality=mod, num_samples=args.simple_samples, seed=args.seed, use_symlinks=use_symlinks
            )

        logger.info("\n✅ 简化数据集生成完成!")
        return 0

    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"比例之和必须为 1.0，当前为 {total_ratio}")
        return 1

    try:
        converter = FREDtoYOLOConverter(
            fred_root=args.fred_root, output_root=args.output_root, split_mode=args.split_mode
        )

        converter.generate_all_splits(
            modality=args.modality,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            use_symlinks=use_symlinks,
        )

        logger.info("\n✅ 所有转换完成！")
        return 0

    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
