#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 YOLOv8 模型结构，确定 Backbone 的层数
"""

from ultralytics import YOLO

def check_model_structure(model_size='n'):
    """检查模型结构"""
    print(f"检查 YOLOv8{model_size} 模型结构...")
    
    # 加载模型
    model = YOLO(f'yolov8{model_size}.pt')
    
    # 打印模型结构
    print("\n模型结构:")
    
    # 检查模型是否是DetectionModel
    if hasattr(model.model, 'model'):
        # 如果是DetectionModel，访问内部的model列表
        model_layers = model.model.model
    else:
        # 否则直接使用model
        model_layers = model.model
    
    for i, layer in enumerate(model_layers):
        layer_type = type(layer).__name__
        print(f"层 {i}: {layer_type}")
        
        # 检查是否有特定的层类型，帮助我们确定 Backbone 的范围
        if 'Detect' in layer_type:
            print(f"  ^^^ 检测头开始于层 {i} ^^^")
    
    # 尝试获取模型的详细结构
    print(f"\n模型总层数: {len(model_layers)}")
    
    # 检查模型配置
    if hasattr(model.model, 'yaml'):
        print("\n模型配置:")
        print(model.model.yaml)
    
    return model

if __name__ == '__main__':
    # 检查不同大小的模型
    for size in ['n', 's', 'm', 'l', 'x']:
        print("\n" + "="*70)
        try:
            model = check_model_structure(size)
        except Exception as e:
            print(f"检查 YOLOv8{size} 时出错: {e}")