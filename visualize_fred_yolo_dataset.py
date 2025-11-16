#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# -*- coding: utf-8 -*-
"""
FRED YOLO 数据集可视化工具.

将 YOLO 格式的标注文件中的边界框绘制到图像上，生成可视化视频序列。
支持单序列或多序列的可视化，可以调整显示参数和输出格式。

主要功能：
1. 读取 YOLO 格式的标注文件
2. 在图像上绘制边界框和类别标签
3. 生成视频序列或单独的图像文件
4. 支持自定义颜色、字体和显示参数
5. 支持多种输出格式

使用方法：
    # 可视化单个序列
    python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1

    # 可视化多个序列
    python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-ids 1,2,3

    # 生成视频文件
    python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1 --output-video

    # 自定义显示参数
    python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1 --bbox-thickness 2 --font-size 1.0
"""

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FREDYOLOVisualizer:
    """FRED YOLO 数据集可视化器."""

    def __init__(
        self,
        dataset_dir,
        modality="rgb",
        bbox_color=(0, 255, 0),
        text_color=(255, 255, 255),
        bbox_thickness=2,
        font_size=1.0,
        show_confidence=False,
    ):
        """初始化可视化器.

        Args:
            dataset_dir: 数据集根目录
            modality: 'rgb' 或 'event'
            bbox_color: 边界框颜色 (B, G, R)
            text_color: 文本颜色 (B, G, R)
            bbox_thickness: 边界框粗细
            font_size: 字体大小比例
            show_confidence: 是否显示置信度（YOLO 格式默认无置信度）
        """
        self.dataset_dir = Path(dataset_dir)
        self.modality = modality
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.bbox_thickness = bbox_thickness
        self.font_size = font_size
        self.show_confidence = show_confidence

        # YOLO 类别定义
        self.class_names = ["drone"]

        # 图像和标签目录
        self.dataset_name = f"fred_{modality}"
        self.images_dir = self.dataset_dir / self.dataset_name / "images"
        self.labels_dir = self.dataset_dir / self.dataset_name / "labels"

        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_thickness = max(1, int(bbox_thickness / 2))

        # 验证数据集目录
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")

        if not self.labels_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")

        logger.info(f"数据集目录: {self.dataset_dir}")
        logger.info(f"模态: {modality}")
        logger.info(f"图像目录: {self.images_dir}")
        logger.info(f"标签目录: {self.labels_dir}")

    def parse_filename(self, filename):
        """解析文件名，提取序列ID和时间戳.

        Args:
            filename: 文件名，格式为 "{序列ID:04d}_{时间戳:.6f}.jpg"

        Returns:
            tuple: (sequence_id, timestamp)
        """
        try:
            base_name = Path(filename).stem
            parts = base_name.split("_")

            if len(parts) >= 2:
                sequence_id = int(parts[0])
                timestamp = float(parts[1])
                return sequence_id, timestamp
        except (ValueError, IndexError) as e:
            logger.warning(f"无法解析文件名: {filename}, 错误: {e}")

        return None, None

    def get_splits_for_sequence(self, sequence_id):
        """获取指定序列ID在所有划分中的图像文件.

        Args:
            sequence_id: 序列ID

        Returns:
            dict: {split: [(timestamp, image_path, label_path), ...]}
        """
        splits = {}

        for split in ["train", "val", "test"]:
            split_images_dir = self.images_dir / split
            split_labels_dir = self.labels_dir / split

            if not split_images_dir.exists():
                continue

            files = []

            # 查找该序列的所有图像
            pattern = f"{sequence_id:04d}_*.{self._get_image_extension()}"
            for image_path in sorted(split_images_dir.glob(pattern)):
                # 解析文件名获取时间戳
                file_seq_id, timestamp = self.parse_filename(image_path.name)

                if file_seq_id is None or file_seq_id != sequence_id:
                    continue

                # 查找对应的标签文件
                # 标准格式应为 {image_name_without_ext}.txt
                label_path = split_labels_dir / f"{image_path.stem}.txt"

                # 检查标签文件是否存在
                if not label_path.exists():
                    # 兼容旧格式：尝试查找 {image_name}.jpg.txt 格式的标签文件
                    label_path = split_labels_dir / f"{image_path.name}.txt"

                # 检查标签文件或.path文件是否存在
                if not label_path.exists():
                    # 尝试查找 .path 文件
                    path_file = split_labels_dir / f"{image_path.stem}.txt.path"
                    if path_file.exists():
                        label_path = path_file
                    else:
                        # 兼容旧格式：尝试查找 {image_name}.jpg.txt.path 格式的路径文件
                        path_file = split_labels_dir / f"{image_path.name}.txt.path"
                        if path_file.exists():
                            label_path = path_file
                        else:
                            logger.warning(f"未找到标签文件: {label_path}")
                            continue

                files.append((timestamp, image_path, label_path))

            if files:
                # 按时间戳排序
                files = sorted(files, key=lambda x: x[0])
                splits[split] = files

        return splits

    def _get_image_extension(self):
        """获取图像文件扩展名."""
        if self.modality == "rgb":
            return "jpg"
        elif self.modality == "event":
            return "png"
        else:
            return "jpg"

    def load_image(self, image_path):
        """加载图像文件.

        Args:
            image_path: 图像路径

        Returns:
            numpy.ndarray: 图像数组
        """
        # 检查是否是符号链接或.path文件
        if image_path.is_symlink():
            # 读取符号链接目标
            actual_path = image_path.resolve()
            if not actual_path.exists():
                logger.error(f"符号链接目标不存在: {image_path} -> {actual_path}")
                return None
            image_path = actual_path
        elif not image_path.exists():
            # 检查是否有.path文件
            path_file = Path(str(image_path) + ".path")
            if path_file.exists():
                try:
                    with open(path_file) as f:
                        actual_path = Path(f.read().strip())
                    if not actual_path.exists():
                        logger.error(f".path文件中路径不存在: {path_file}")
                        return None
                    image_path = actual_path
                except Exception as e:
                    logger.error(f"读取.path文件失败: {path_file}, 错误: {e}")
                    return None
            else:
                logger.error(f"图像文件不存在: {image_path}")
                return None

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"读取图像失败: {image_path}, 错误: {e}")
            return None

    def load_labels(self, label_path):
        """加载YOLO格式的标签文件.

        Args:
            label_path: 标签文件路径

        Returns:
            list: [(class_id, x_center, y_center, width, height), ...]
        """
        labels = []

        # 检查是否是.path文件
        if label_path.suffix == ".path":
            try:
                with open(label_path) as f:
                    actual_path = Path(f.read().strip())
                label_path = actual_path
            except Exception as e:
                logger.error(f"读取.path文件失败: {label_path}, 错误: {e}")
                return labels

        if not label_path.exists():
            return labels

        try:
            with open(label_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            # 可选的置信度
                            confidence = float(parts[5]) if len(parts) > 5 else 1.0

                            labels.append((class_id, x_center, y_center, width, height, confidence))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"标签文件 {label_path} 第 {line_num} 行解析失败: {e}")
                        continue
        except Exception as e:
            logger.error(f"读取标签文件失败: {label_path}, 错误: {e}")

        return labels

    def yolo_to_bbox(self, x_center, y_center, width, height, img_width, img_height):
        """将YOLO格式的边界框转换为像素坐标.

        Args:
            x_center, y_center: 归一化的中心点坐标
            width, height: 归一化的宽高
            img_width, img_height: 图像尺寸

        Returns:
            tuple: (x1, y1, x2, y2) 像素坐标
        """
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        return x1, y1, x2, y2

    def draw_bbox(self, image, bbox, class_id, confidence=1.0):
        """在图像上绘制边界框.

        Args:
            image: 图像数组
            bbox: (x1, y1, x2, y2) 边界框坐标
            class_id: 类别ID
            confidence: 置信度

        Returns:
            numpy.ndarray: 绘制了边界框的图像
        """
        x1, y1, x2, y2 = bbox

        # 获取类别名称
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)

        # 准备标签文本
        label = class_name
        if self.show_confidence and confidence < 1.0:
            label += f" {confidence:.2f}"

        # 计算文本尺寸
        font_scale = self.font_size
        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, font_scale, self.font_thickness)

        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), self.bbox_color, -1)

        # 绘制文本
        cv2.putText(image, label, (x1, y1 - baseline - 2), self.font, font_scale, self.text_color, self.font_thickness)

        return image

    def visualize_frame(self, image_path, label_path, output_path=None):
        """可视化单帧图像.

        Args:
            image_path: 图像路径
            label_path: 标签路径
            output_path: 输出路径（可选）

        Returns:
            numpy.ndarray: 可视化后的图像
        """
        # 加载图像
        image = self.load_image(image_path)
        if image is None:
            return None

        # 加载标签
        labels = self.load_labels(label_path)

        # 获取图像尺寸
        img_height, img_width = image.shape[:2]

        # 绘制所有边界框
        for class_id, x_center, y_center, width, height, confidence in labels:
            bbox = self.yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            image = self.draw_bbox(image, bbox, class_id, confidence)

        # 保存输出图像
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)

        return image

    def visualize_sequence(
        self, sequence_id, output_dir=None, save_frames=False, output_video=None, fps=30, show_progress=True
    ):
        """可视化整个序列.

        Args:
            sequence_id: 序列ID
            output_dir: 输出目录
            save_frames: 是否保存单独的帧
            output_video: 输出视频路径
            fps: 视频帧率
            show_progress: 是否显示进度条

        Returns:
            dict: 统计信息
        """
        logger.info(f"可视化序列 {sequence_id}...")

        # 获取序列的所有帧
        splits = self.get_splits_for_sequence(sequence_id)

        if not splits:
            logger.warning(f"序列 {sequence_id} 没有找到任何帧")
            return {}

        # 创建输出目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if save_frames:
                frames_dir = output_dir / f"sequence_{sequence_id:04d}_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)

        # 统计信息
        stats = {"total_frames": 0, "processed_frames": 0, "total_bboxes": 0, "splits": list(splits.keys())}

        # 视频写入器
        video_writer = None

        try:
            # 处理每个划分的帧
            for split, frames in splits.items():
                logger.info(f"处理 {split} 划分: {len(frames)} 帧")

                # 创建进度条
                frame_iter = tqdm(frames, desc=f"处理 {split} 帧", disable=not show_progress)

                for timestamp, image_path, label_path in frame_iter:
                    stats["total_frames"] += 1

                    # 可视化帧
                    if save_frames and output_dir:
                        frame_output_path = frames_dir / f"{split}_{timestamp:.6f}.jpg"
                    else:
                        frame_output_path = None

                    vis_image = self.visualize_frame(image_path, label_path, frame_output_path)

                    if vis_image is None:
                        continue

                    stats["processed_frames"] += 1

                    # 统计边界框数量
                    labels = self.load_labels(label_path)
                    stats["total_bboxes"] += len(labels)

                    # 写入视频
                    if output_video:
                        if video_writer is None:
                            height, width = vis_image.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

                        video_writer.write(vis_image)

            # 释放视频写入器
            if video_writer is not None:
                video_writer.release()

                if output_video:
                    logger.info(f"视频已保存: {output_video}")

        except Exception as e:
            logger.error(f"可视化序列 {sequence_id} 时出错: {e}")
            if video_writer is not None:
                video_writer.release()

        logger.info(f"序列 {sequence_id} 可视化完成:")
        logger.info(f"  总帧数: {stats['total_frames']}")
        logger.info(f"  处理帧数: {stats['processed_frames']}")
        logger.info(f"  总边界框: {stats['total_bboxes']}")
        logger.info(f"  划分: {', '.join(stats['splits'])}")

        return stats

    def visualize_multiple_sequences(
        self, sequence_ids, output_dir=None, save_frames=False, output_video_dir=None, fps=30, show_progress=True
    ):
        """可视化多个序列.

        Args:
            sequence_ids: 序列ID列表
            output_dir: 输出目录
            save_frames: 是否保存单独的帧
            output_video_dir: 输出视频目录
            fps: 视频帧率
            show_progress: 是否显示进度条

        Returns:
            dict: 所有序列的统计信息
        """
        logger.info(f"可视化 {len(sequence_ids)} 个序列...")

        # 创建输出目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if output_video_dir:
            output_video_dir = Path(output_video_dir)
            output_video_dir.mkdir(parents=True, exist_ok=True)

        # 所有序列的统计信息
        all_stats = {}

        # 处理每个序列
        for i, sequence_id in enumerate(sequence_ids):
            logger.info(f"\n处理序列 {i + 1}/{len(sequence_ids)}: {sequence_id}")

            # 设置输出路径
            seq_output_dir = output_dir / f"sequence_{sequence_id:04d}" if output_dir else None
            seq_output_video = None

            if output_video_dir:
                seq_output_video = output_video_dir / f"sequence_{sequence_id:04d}.mp4"

            # 可视化序列
            stats = self.visualize_sequence(
                sequence_id, seq_output_dir, save_frames, seq_output_video, fps, show_progress
            )

            if stats:
                all_stats[sequence_id] = stats

        # 打印总体统计
        logger.info(f"\n{'=' * 70}")
        logger.info("所有序列可视化完成!")
        logger.info(f"{'=' * 70}")

        total_frames = sum(s.get("total_frames", 0) for s in all_stats.values())
        total_processed = sum(s.get("processed_frames", 0) for s in all_stats.values())
        total_bboxes = sum(s.get("total_bboxes", 0) for s in all_stats.values())

        logger.info(f"总序列数: {len(all_stats)}")
        logger.info(f"总帧数: {total_frames}")
        logger.info(f"处理帧数: {total_processed}")
        logger.info(f"总边界框: {total_bboxes}")

        return all_stats


def parse_sequence_ids(sequence_ids_str):
    """解析序列ID字符串.

    Args:
        sequence_ids_str: 序列ID字符串，如 "1,2,3" 或 "1-3"

    Returns:
        list: 序列ID列表
    """
    sequence_ids = []

    for part in sequence_ids_str.split(","):
        part = part.strip()
        if "-" in part:
            # 处理范围，如 "1-3"
            try:
                start, end = map(int, part.split("-"))
                sequence_ids.extend(range(start, end + 1))
            except ValueError:
                logger.warning(f"无效的序列ID范围: {part}")
        else:
            # 处理单个ID
            try:
                sequence_ids.append(int(part))
            except ValueError:
                logger.warning(f"无效的序列ID: {part}")

    return sorted(list(set(sequence_ids)))


def main():
    parser = argparse.ArgumentParser(
        description="FRED YOLO 数据集可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化单个序列
  python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1
  
  # 可视化多个序列
  python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-ids 1,2,3
  
  # 生成视频文件
  python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1 --output-video
  
  # 保存单独的帧
  python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1 --save-frames
  
  # 自定义显示参数
  python visualize_fred_yolo_dataset.py --dataset-dir datasets/fred_yolo --sequence-id 1 --bbox-color 0,255,0 --font-size 1.2
        """,
    )

    # 数据集参数
    parser.add_argument("--dataset-dir", type=str, required=True, help="数据集根目录")
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "event"], help="数据模态")

    # 序列选择参数
    parser.add_argument("--sequence-id", type=int, help="要可视化的单个序列ID")
    parser.add_argument("--sequence-ids", type=str, help='要可视化的多个序列ID，如 "1,2,3" 或 "1-3"')

    # 输出参数
    parser.add_argument("--output-dir", type=str, default="visualization_output", help="输出目录")
    parser.add_argument("--save-frames", action="store_true", help="保存单独的帧图像")
    parser.add_argument("--output-video", action="store_true", help="生成视频文件")
    parser.add_argument("--output-video-dir", type=str, help="视频输出目录（用于多序列）")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")

    # 显示参数
    parser.add_argument("--bbox-color", type=str, default="0,255,0", help="边界框颜色 (B,G,R)")
    parser.add_argument("--text-color", type=str, default="255,255,255", help="文本颜色 (B,G,R)")
    parser.add_argument("--bbox-thickness", type=int, default=2, help="边界框粗细")
    parser.add_argument("--font-size", type=float, default=1.0, help="字体大小比例")
    parser.add_argument("--show-confidence", action="store_true", help="显示置信度（如果有）")

    # 其他参数
    parser.add_argument("--show-progress", action="store_true", default=True, help="显示进度条")

    args = parser.parse_args()

    # 验证参数
    if not args.sequence_id and not args.sequence_ids:
        logger.error("必须指定 --sequence-id 或 --sequence-ids")
        return 1

    # 解析颜色
    try:
        bbox_color = tuple(map(int, args.bbox_color.split(",")))
        text_color = tuple(map(int, args.text_color.split(",")))
    except ValueError:
        logger.error("颜色格式错误，应为 'R,G,B'")
        return 1

    # 创建可视化器
    try:
        visualizer = FREDYOLOVisualizer(
            dataset_dir=args.dataset_dir,
            modality=args.modality,
            bbox_color=bbox_color,
            text_color=text_color,
            bbox_thickness=args.bbox_thickness,
            font_size=args.font_size,
            show_confidence=args.show_confidence,
        )
    except Exception as e:
        logger.error(f"创建可视化器失败: {e}")
        return 1

    # 确定序列ID
    if args.sequence_id:
        sequence_ids = [args.sequence_id]
    else:
        sequence_ids = parse_sequence_ids(args.sequence_ids)

    if not sequence_ids:
        logger.error("没有有效的序列ID")
        return 1

    # 可视化
    try:
        if len(sequence_ids) == 1:
            # 单个序列
            sequence_id = sequence_ids[0]

            # 设置输出路径
            output_dir = args.output_dir
            output_video = None

            if args.output_video:
                output_video = Path(args.output_dir) / f"sequence_{sequence_id:04d}.mp4"

            # 可视化序列
            visualizer.visualize_sequence(
                sequence_id, output_dir, args.save_frames, output_video, args.fps, args.show_progress
            )
        else:
            # 多个序列
            visualizer.visualize_multiple_sequences(
                sequence_ids,
                args.output_dir,
                args.save_frames,
                args.output_video_dir or Path(args.output_dir) / "videos",
                args.fps,
                args.show_progress,
            )

        logger.info("\n✅ 可视化完成!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ 可视化失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
