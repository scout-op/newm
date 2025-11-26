"""
GraphEvaluator - 统一的图评价接口
整合所有LaneDiffusion指标的计算
"""

import numpy as np
import networkx as nx
from .geotopo import Evaluator as GeoTopoEvaluator
from .topo_metrics import calc_iou, calc_apls, calc_sda
from .topo_metrics import nx_to_geo_topo_format
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import random
import warnings
from shapely.geometry import LineString
import uuid

random.seed(0)


class GraphEvaluator():
    """
    图评价器 - 计算LaneDiffusion的9个指标
    
    Args:
        radius: GEO/TOPO匹配半径
        interp_dist: 插值距离
        prop_dist: 传播距离
        area_size: 渲染区域大小 [宽, 高]
        lane_width: 渲染车道宽度
        gmode: 图模式 ('direct' 或 'undirect')
    """

    def __init__(self, radius=8, interp_dist=2, prop_dist=400, area_size=[30, 60], lane_width=1, gmode='direct'):
        self.radius = radius
        self.interp_dist = interp_dist
        self.prop_dist = prop_dist
        self.area_size = area_size
        self.lane_width = lane_width
        self.gmode = gmode

    def evaluate_graph(self, graph_gt: nx.DiGraph, graph_pred: nx.DiGraph, verbose=False):
        """
        评价预测图与ground truth图
        
        Args:
            graph_gt: ground truth图（networkx.DiGraph）
            graph_pred: 预测图（networkx.DiGraph）
            verbose: 是否打印详细信息
        
        Returns:
            包含9个指标的字典
        """
        # 为边添加长度属性
        graph_gt = assign_edge_lengths(graph_gt)
        graph_pred = assign_edge_lengths(graph_pred)

        # 计算IoU
        iou = calc_iou(graph_gt, graph_pred, area_size=self.area_size, lane_width=self.lane_width)
        
        # 计算SDA（分叉点检测准确度）
        sda = calc_sda(graph_gt, graph_pred)

        # 准备APLS计算所需的图格式
        graph_gt_for_apls = prepare_graph_apls(graph_gt)
        graph_pred_for_apls = prepare_graph_apls(graph_pred)

        # 计算APLS指标
        try:
            apls = calc_apls(graph_gt_for_apls, graph_pred_for_apls)
        except Exception as e:
            apls = 0
            warnings.warn("Error calculating APLS metric: {}.".format(e))
        
        if verbose:
            print("iou : ", iou)
            print("sda : ", sda)
            print("apls : ", apls)

        # 转换为GEO/TOPO计算所需的格式
        graph_gt_ = nx_to_geo_topo_format(graph_gt)
        graph_pred_ = nx_to_geo_topo_format(graph_pred)

        # 创建GEO/TOPO评价器
        evaluator = GeoTopoEvaluator(graph_gt_, graph_pred_, self.interp_dist, self.prop_dist, self.gmode)
        
        # 计算GEO和TOPO指标
        (geo_precision, 
        geo_recall, 
        topo_precision, 
        topo_recall,
        geo_f1,
        topo_f1,
        jtopo_f1) = \
        evaluator.topoMetric(thr=self.radius, verbose=verbose)

        # 返回所有指标
        metrics_dict = {
            'Graph IoU': iou,
            'APLS': apls,
            'GEO Precision': geo_precision,
            'GEO Recall': geo_recall,
            'GEO F1': geo_f1,
            'TOPO Precision': topo_precision,
            'TOPO Recall': topo_recall,
            'TOPO F1': topo_f1,
            'JTOPO F1': jtopo_f1,
            'SDA': sda
        }

        return metrics_dict


def truncated_uuid4():
    """生成截断的UUID"""
    return str(int(uuid.uuid4()))[0:6]


def prepare_graph_apls(g):
    """
    准备APLS计算所需的图格式
    
    Args:
        g: networkx图
    
    Returns:
        准备好的图
    """
    # 重新标注节点，避免名称冲突
    g = nx.relabel_nodes(g, {n: str(g.nodes[n]['pos']) + truncated_uuid4() for n in g.nodes()})

    # 转换为无向图
    g = nx.to_undirected(g)

    # 添加x, y坐标属性
    for n, d in g.nodes(data=True):
        d['x'] = d['pos'][0]
        d['y'] = d['pos'][1]

    # 添加几何和长度属性
    for u, v, d in g.edges(data=True):
        d['geometry'] = LineString([(g.nodes[u]['x'], g.nodes[u]['y']),
                                    (g.nodes[v]['x'], g.nodes[v]['y'])])
        d['length'] = d['geometry'].length

    return g


def assign_edge_lengths(G):
    """
    为图的每条边计算并添加长度属性
    
    Args:
        G: networkx图
    
    Returns:
        添加了长度属性的图
    """
    for u, v, d in G.edges(data=True):
        d['length'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
    return G
