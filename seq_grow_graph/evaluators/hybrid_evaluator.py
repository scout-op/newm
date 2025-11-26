"""
混合评价器 - 同时计算SeqGrowGraph和LaneDiffusion的所有指标
"""

import numpy as np
from typing import Dict, List
import warnings

from ..adapters.graph_converter import eval_seq2graph_to_networkx
from .lanediffusion import GraphEvaluator


class HybridEvaluator:
    """
    混合评价器 - 统一评价接口
    
    同时计算:
    1. SeqGrowGraph原有指标: Landmark, Reachability
    2. LaneDiffusion新增指标: GEO F1, TOPO F1, IoU, APLS等
    
    Args:
        landmark_thresholds: Landmark指标的阈值列表
        reach_thresholds: Reachability指标的阈值列表
        lanediff_radius: LaneDiffusion GEO/TOPO指标匹配半径
        lanediff_interp_dist: 插值距离
        lanediff_prop_dist: 传播距离
        lanediff_area_size: 渲染区域大小
        lanediff_lane_width: 渲染车道宽度
        conversion_mode: 图转换模式 ('simple' or 'interpolated')
        interp_points: 插值点数量
    """
    
    def __init__(
        self,
        landmark_thresholds=[1, 3, 5],
        reach_thresholds=[1, 2, 3],
        lanediff_radius=5,
        lanediff_interp_dist=2.5,
        lanediff_prop_dist=80,
        lanediff_area_size=[30, 60],
        lanediff_lane_width=1,
        conversion_mode='simple',
        interp_points=20
    ):
        self.landmark_thresholds = landmark_thresholds
        self.reach_thresholds = reach_thresholds
        self.conversion_mode = conversion_mode
        self.interp_points = interp_points
        
        # 初始化LaneDiffusion评价器
        self.graph_evaluator = GraphEvaluator(
            radius=lanediff_radius,
            interp_dist=lanediff_interp_dist,
            prop_dist=lanediff_prop_dist,
            area_size=lanediff_area_size,
            lane_width=lanediff_lane_width,
            gmode='direct'
        )
    
    def evaluate_lanediffusion_single(
        self,
        gt_eval_graph,
        pred_eval_graph,
        verbose=False,
        pc_range=None,
        dx=None
    ) -> Dict[str, float]:
        """
        对单个样本计算LaneDiffusion指标
        
        Args:
            gt_eval_graph: ground truth的EvalSeq2Graph对象
            pred_eval_graph: 预测的EvalSeq2Graph对象
            verbose: 是否打印详细信息
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            dx: 网格分辨率
        
        Returns:
            包含9个指标的字典
        """
        try:
            # 转换为networkx格式
            gt_graph = eval_seq2graph_to_networkx(
                gt_eval_graph,
                mode=self.conversion_mode,
                interp_points=self.interp_points,
                pc_range=pc_range,
                dx=dx
            )
            pred_graph = eval_seq2graph_to_networkx(
                pred_eval_graph,
                mode=self.conversion_mode,
                interp_points=self.interp_points,
                pc_range=pc_range,
                dx=dx
            )
            
            # 检查图是否为空
            if len(gt_graph.nodes()) == 0 or len(pred_graph.nodes()) == 0:
                if verbose:
                    print(f"Warning: Empty graph detected. GT nodes: {len(gt_graph.nodes())}, Pred nodes: {len(pred_graph.nodes())}")
                return self._get_zero_metrics()
            
            # 计算所有LaneDiffusion指标
            metrics = self.graph_evaluator.evaluate_graph(
                gt_graph,
                pred_graph,
                verbose=verbose
            )
            
            # 添加前缀以区分来源
            prefixed_metrics = {}
            for key, value in metrics.items():
                prefixed_metrics[f'LaneDiff/{key}'] = value
            
            return prefixed_metrics
            
        except Exception as e:
            warnings.warn(f"Error evaluating LaneDiffusion metrics: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return self._get_zero_metrics()
    
    def evaluate_lanediffusion_batch(
        self,
        gt_graphs_list,
        pred_graphs_list,
        verbose=False,
        pc_range=None,
        dx=None
    ) -> Dict[str, float]:
        """
        批量计算LaneDiffusion指标并求平均
        
        Args:
            gt_graphs_list: GT图列表
            pred_graphs_list: 预测图列表
            verbose: 是否打印详细信息
            pc_range: 点云范围
            dx: 网格分辨率
        
        Returns:
            平均后的指标字典
        """
        assert len(gt_graphs_list) == len(pred_graphs_list), \
            "GT and prediction lists must have the same length"
        
        all_metrics = []
        
        for i, (gt_graph, pred_graph) in enumerate(zip(gt_graphs_list, pred_graphs_list)):
            metrics = self.evaluate_lanediffusion_single(
                gt_graph,
                pred_graph,
                verbose=False,
                pc_range=pc_range,
                dx=dx
            )
            all_metrics.append(metrics)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(gt_graphs_list)} samples")
        
        # 计算平均值
        avg_metrics = {}
        metric_keys = all_metrics[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if len(values) > 0:
                avg_metrics[key] = np.nanmean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def _get_zero_metrics(self) -> Dict[str, float]:
        """返回全零的指标字典（用于错误情况）"""
        return {
            'LaneDiff/Graph IoU': 0.0,
            'LaneDiff/APLS': 0.0,
            'LaneDiff/GEO Precision': 0.0,
            'LaneDiff/GEO Recall': 0.0,
            'LaneDiff/GEO F1': 0.0,
            'LaneDiff/TOPO Precision': 0.0,
            'LaneDiff/TOPO Recall': 0.0,
            'LaneDiff/TOPO F1': 0.0,
            'LaneDiff/JTOPO F1': 0.0,
            'LaneDiff/SDA': 0.0
        }
    
    def format_results(self, metrics: Dict[str, float], logger=None) -> str:
        """
        格式化输出结果
        
        Args:
            metrics: 指标字典
            logger: 日志记录器（可选）
        
        Returns:
            格式化的字符串
        """
        lines = [
            "=" * 50,
            "LaneDiffusion Metrics",
            "=" * 50,
            f"GEO Precision:  {metrics.get('LaneDiff/GEO Precision', 0):.3f}",
            f"GEO Recall:     {metrics.get('LaneDiff/GEO Recall', 0):.3f}",
            f"GEO F1:         {metrics.get('LaneDiff/GEO F1', 0):.3f}",
            "-" * 50,
            f"TOPO Precision: {metrics.get('LaneDiff/TOPO Precision', 0):.3f}",
            f"TOPO Recall:    {metrics.get('LaneDiff/TOPO Recall', 0):.3f}",
            f"TOPO F1:        {metrics.get('LaneDiff/TOPO F1', 0):.3f}",
            "-" * 50,
            f"JTOPO F1:       {metrics.get('LaneDiff/JTOPO F1', 0):.3f}",
            f"APLS:           {metrics.get('LaneDiff/APLS', 0):.3f}",
            f"IoU:            {metrics.get('LaneDiff/Graph IoU', 0):.3f}",
            f"SDA:            {metrics.get('LaneDiff/SDA', 0):.3f}",
            "=" * 50
        ]
        
        result_str = "\n".join(lines)
        
        if logger:
            for line in lines:
                logger.info(line)
        
        return result_str
