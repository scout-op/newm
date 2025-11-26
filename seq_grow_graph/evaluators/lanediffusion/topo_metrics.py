"""
图评价指标计算函数
包含IoU, SDA, APLS等指标
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import warnings
from scipy.spatial.distance import cdist
from .apls import execute_apls
from .geotopo import f1_score


def filter_graph(target, source, threshold=100):
    """
    过滤图中距离目标图太远的节点
    """
    if len(source.nodes()) == 0:
        return source

    pos_target = np.array([list(target.nodes[n]['pos']) for n in target.nodes()])
    pos_source = np.array([list(source.nodes[n]['pos']) for n in source.nodes()])

    distance_matrix = cdist(pos_target, pos_source)

    source_ = source.copy(as_view=False)
    is_close_to_target = np.min(distance_matrix, axis=0) < threshold
    for i, n in enumerate(list(source_.nodes())):
        if not is_close_to_target[i]:
            if n in source.nodes():
                source.remove_node(n)

    return source


def calc_sda(graph_gt, graph_pred, threshold=1):
    """
    计算分叉点检测准确度 (Split Detection Accuracy)
    
    Args:
        graph_gt: ground truth图
        graph_pred: 预测图
        threshold: 匹配阈值
    
    Returns:
        SDA F1分数
    """
    split_point_positions_gt = []
    split_point_positions_pred = []

    # 提取GT中的分叉点
    for n in graph_gt.nodes():
        if graph_gt.out_degree(n) >= 2 or graph_gt.in_degree(n) >= 2:
            split_point_positions_gt.append(graph_gt.nodes[n]['pos'])

    # 提取预测中的分叉点
    for n in graph_pred.nodes():
        if graph_pred.out_degree(n) >= 2 or graph_pred.in_degree(n) >= 2:
            split_point_positions_pred.append(graph_pred.nodes[n]['pos'])

    if len(split_point_positions_gt) == 0:
        return np.nan

    if len(split_point_positions_pred) == 0:
        return 0.0

    # 构建代价矩阵
    split_point_positions_gt = np.array(split_point_positions_gt)
    split_point_positions_pred = np.array(split_point_positions_pred)

    cost_matrix = np.linalg.norm(
        split_point_positions_gt[:, None, :] - split_point_positions_pred[None, :, :], 
        axis=-1
    )

    # 匈牙利算法匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = np.sum(cost_matrix[row_ind, col_ind] < threshold)
    fp = len(split_point_positions_pred) - tp
    fn = len(split_point_positions_gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = f1_score(precision, recall)

    return F1


def calc_apls(g_gt, g_pred):
    """
    计算APLS（Average Path Length Similarity）指标
    
    Args:
        g_gt: ground truth图
        g_pred: 预测图
    
    Returns:
        APLS分数
    """
    g_gt = [g_gt]
    g_pred = [g_pred]

    apls_dict = execute_apls(g_gt, g_pred, verbose=False)

    return apls_dict['APLS']


def render_graph(graph, imsize=[256, 256], width=10):
    """
    将图渲染为图像
    
    Args:
        graph: networkx图
        imsize: 图像尺寸
        width: 线宽
    
    Returns:
        渲染的图像
    """
    im = np.zeros(imsize).astype(np.uint8)

    for e in graph.edges():
        start = graph.nodes[e[0]]['pos']
        end = graph.nodes[e[1]]['pos']
        x1 = int(start[0])
        y1 = int(start[1])
        x2 = int(end[0])
        y2 = int(end[1])
        cv2.line(im, (x1, y1), (x2, y2), 255, width)
    
    return im


def calc_iou(graph_gt, graph_pred, area_size=[256, 256], lane_width=10):
    """
    计算两个图的IoU（Intersection over Union）
    
    Args:
        graph_gt: ground truth图
        graph_pred: 预测图
        area_size: 渲染区域大小
        lane_width: 车道宽度
    
    Returns:
        IoU分数
    """
    render_gt = render_graph(graph_gt, imsize=area_size, width=lane_width)
    render_pred = render_graph(graph_pred, imsize=area_size, width=lane_width)

    # 计算IoU
    intersection = np.logical_and(render_gt, render_pred)
    union = np.logical_or(render_gt, render_pred)
    iou = np.sum(intersection) / (1e-8 + np.sum(union))

    return iou


def nx_to_geo_topo_format(nx_graph):
    """
    将networkx图转换为GEO/TOPO指标计算所需的格式
    
    Args:
        nx_graph: networkx图
    
    Returns:
        字典格式的图: {节点: [邻居节点列表]}
    """
    neighbors = {}

    for e in nx_graph.edges():
        x1 = nx_graph.nodes[e[0]]['pos'][0]
        y1 = nx_graph.nodes[e[0]]['pos'][1]
        x2 = nx_graph.nodes[e[1]]['pos'][0]
        y2 = nx_graph.nodes[e[1]]['pos'][1]

        k1 = (int(x1 * 10), int(y1 * 10))
        k2 = (int(x2 * 10), int(y2 * 10))

        if k1 not in neighbors:
            neighbors[k1] = []

        if k2 not in neighbors[k1]:
            neighbors[k1].append(k2)

    return neighbors
