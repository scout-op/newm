"""
LaneDiffusion指标集成使用示例

演示如何使用混合评价器计算LaneDiffusion指标
"""

import sys
import json
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from seq_grow_graph.evaluators import HybridEvaluator
from seq_grow_graph.adapters import eval_seq2graph_to_networkx


def example_single_evaluation():
    """
    示例1: 评价单个样本
    """
    print("=" * 60)
    print("示例1: 单样本评价")
    print("=" * 60)
    
    # 初始化混合评价器
    evaluator = HybridEvaluator(
        lanediff_radius=5,
        lanediff_interp_dist=2.5,
        lanediff_prop_dist=80,
        conversion_mode='simple'
    )
    
    # 这里需要实际的EvalSeq2Graph对象
    # 示例：假设已经从数据集加载
    # gt_graph = load_gt_graph(sample_id)
    # pred_graph = load_prediction(sample_id)
    
    # metrics = evaluator.evaluate_lanediffusion_single(
    #     gt_graph,
    #     pred_graph,
    #     verbose=True
    # )
    
    # # 打印结果
    # print(evaluator.format_results(metrics))
    
    print("(需要实际的数据才能运行)")


def example_batch_evaluation():
    """
    示例2: 批量评价
    """
    print("\n" + "=" * 60)
    print("示例2: 批量评价")
    print("=" * 60)
    
    evaluator = HybridEvaluator(
        lanediff_radius=5,
        conversion_mode='simple'
    )
    
    # 示例：批量加载数据
    # gt_graphs = [load_gt_graph(i) for i in range(100)]
    # pred_graphs = [load_prediction(i) for i in range(100)]
    
    # metrics = evaluator.evaluate_lanediffusion_batch(
    #     gt_graphs,
    #     pred_graphs,
    #     verbose=True
    # )
    
    # print(evaluator.format_results(metrics))
    
    print("(需要实际的数据才能运行)")


def example_with_nuscenes_metric():
    """
    示例3: 集成到NuScenesReachMetric中使用
    """
    print("\n" + "=" * 60)
    print("示例3: 在评价流程中使用")
    print("=" * 60)
    
    print("""
    在配置文件中启用LaneDiffusion指标:
    
    val_evaluator = dict(
        type='NuScenesReachMetric',
        metric=['ar_reach'],
        landmark_thresholds=[1, 2, 3, 4, 5],
        reach_thresholds=[1, 2, 3, 4, 5],
        
        # 新增：LaneDiffusion配置
        enable_lanediffusion=True,
        lanediff_config=dict(
            lanediff_radius=5,
            lanediff_interp_dist=2.5,
            lanediff_prop_dist=80,
            conversion_mode='simple'
        )
    )
    
    然后运行评价:
    python tools/test.py configs/xxx.py checkpoints/xxx.pth
    """)


def example_direct_networkx_usage():
    """
    示例4: 直接使用networkx进行评价
    """
    print("\n" + "=" * 60)
    print("示例4: 直接使用networkx图")
    print("=" * 60)
    
    import networkx as nx
    from seq_grow_graph.evaluators.lanediffusion import GraphEvaluator
    
    # 创建简单的示例图
    gt_graph = nx.DiGraph()
    gt_graph.add_node(0, pos=(0, 0))
    gt_graph.add_node(1, pos=(10, 0))
    gt_graph.add_node(2, pos=(10, 10))
    gt_graph.add_edge(0, 1)
    gt_graph.add_edge(1, 2)
    
    pred_graph = nx.DiGraph()
    pred_graph.add_node(0, pos=(0, 1))  # 稍有偏差
    pred_graph.add_node(1, pos=(10, 1))
    pred_graph.add_node(2, pos=(10, 11))
    pred_graph.add_edge(0, 1)
    pred_graph.add_edge(1, 2)
    
    # 直接评价
    evaluator = GraphEvaluator(radius=5)
    metrics = evaluator.evaluate_graph(gt_graph, pred_graph, verbose=True)
    
    print("\n评价结果:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    example_single_evaluation()
    example_batch_evaluation()
    example_with_nuscenes_metric()
    example_direct_networkx_usage()
