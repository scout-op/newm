"""
图转换器 - 将SeqGrowGraph的EvalSeq2Graph转换为networkx.DiGraph
这是集成LaneDiffusion指标的核心模块
"""

import numpy as np
import networkx as nx
from typing import Optional, Tuple


def sample_bezier_curve(coeffs, num_points=20):
    """
    对二次贝塞尔曲线进行采样
    
    Args:
        coeffs: 贝塞尔控制点坐标 (2,) - 只包含P1控制点
        num_points: 采样点数量
    
    Returns:
        采样点数组 shape (num_points, 2)
    """
    t_values = np.linspace(0, 1, num_points)
    points = []
    
    # 注意：这里coeffs是中间控制点P1的坐标
    # 贝塞尔曲线: B(t) = (1-t)²P0 + 2(1-t)t*P1 + t²P2
    # 但在实际使用中，可能只需要线性插值
  
    for t in t_values:
        # 简化版本：线性插值
        # point = start_pos * (1-t) + end_pos * t
        # 这里返回采样的t值，实际使用时需要起点和终点
        points.append(t)
    
    return np.array(points)


def get_node_key(node_coords):
    """
    生成节点的唯一标识键
    
    Args:
        node_coords: 节点坐标 (x, y)
    
    Returns:
        节点键 (用于去重)
    """
    # 保留2位小数以容忍微小误差
    return (round(node_coords[0], 2), round(node_coords[1], 2))


def eval_seq2graph_to_networkx(
    eval_graph,
    mode='simple',
    interp_points=20,
    pc_range=None,
    dx=None
):
    """
    将EvalSeq2Graph转换为networkx.DiGraph
    
    Args:
        eval_graph: SeqGrowGraph的EvalSeq2Graph对象
        mode: 转换模式
            - 'simple': 只添加节点和边，不插值
            - 'interpolated': 沿贝塞尔曲线插值添加中间点
        interp_points: 插值模式下每条边的采样点数
        pc_range: 点云范围，用于坐标归一化 [x_min, y_min, z_min, x_max, y_max, z_max]
        dx: 网格分辨率 [dx, dy, dz] 或 float
    
    Returns:
        nx.DiGraph: 可用于GraphEvaluator的networkx图
    """
   
    graph = nx.DiGraph()
    
    if not hasattr(eval_graph, 'graph_nodelist'):
        # 空图
        return graph
    
    # 坐标转换辅助函数
    def transform_coord(x, y):
        if pc_range is not None and dx is not None:
            # 处理 dx 可能是数组的情况，只取前两维
            if hasattr(dx, '__len__'):
                dx_val = float(dx[0])
                dy_val = float(dx[1]) if len(dx) > 1 else float(dx[0])
            else:
                dx_val = float(dx)
                dy_val = float(dx)
            
            # 处理 pc_range 可能是数组的情况
            if hasattr(pc_range, '__len__'):
                x_min = float(pc_range[0])
                y_min = float(pc_range[1])
            else:
                x_min = float(pc_range)
                y_min = float(pc_range)
            
            x = float(x) * dx_val + x_min
            y = float(y) * dy_val + y_min
        return float(x), float(y)
    
    node_mapping = {}  # EvalBzNode对象 -> networkx节点ID
    node_id_counter = 0
    
    # 调试：检查第一个节点的结构
    if len(eval_graph.graph_nodelist) > 0 and pc_range is not None:
        first_node = eval_graph.graph_nodelist[0]
        print(f"\n[DEBUG graph_converter] First node inspection:")
        print(f"  Type: {type(first_node)}")
        print(f"  Has 'coord': {hasattr(first_node, 'coord')}")
        if hasattr(first_node, 'coord'):
            print(f"  coord value: {first_node.coord}")
            print(f"  coord type: {type(first_node.coord)}")
        print(f"  Has 'childs': {hasattr(first_node, 'childs')}")
        if hasattr(first_node, 'childs'):
            print(f"  childs count: {len(first_node.childs)}")
        # 列出所有属性
        print(f"  All attributes: {[attr for attr in dir(first_node) if not attr.startswith('_')][:10]}")
    
    # 第一步：添加所有节点
    nodes_skipped = 0
    nodes_added = 0
    
    for idx, eval_node in enumerate(eval_graph.graph_nodelist):
        if not hasattr(eval_node, 'coord'):
            nodes_skipped += 1
            if idx == 0:
                print(f"[DEBUG] Node {idx} has no 'coord' attribute")
            continue
            
        # 获取节点坐标 (网格坐标)
        coord = eval_node.coord
        # 支持 list, tuple 和 numpy.ndarray
        try:
            if hasattr(coord, '__len__') and len(coord) >= 2:
                x, y = float(coord[0]), float(coord[1])
            else:
                nodes_skipped += 1
                if idx == 0:
                    print(f"[DEBUG] Node {idx} coord too short: {coord}")
                continue
        except (TypeError, IndexError, ValueError) as e:
            nodes_skipped += 1
            if idx == 0:
                print(f"[DEBUG] Node {idx} coord conversion failed: {coord}, error: {e}")
            continue
        
        # 调试第一个节点的坐标转换
        if idx == 0 and pc_range is not None:
            print(f"\n[DEBUG] First node coordinate transformation:")
            print(f"  Grid coord: ({x}, {y})")
        
        # 坐标转换
        x, y = transform_coord(x, y)
        
        if idx == 0 and pc_range is not None:
            print(f"  Physical coord: ({x}, {y})")
            print(f"  Node key: {get_node_key((x, y))}")
        
        # 创建节点键（用于去重）
        node_key = get_node_key((x, y))
        
        # 如果节点已存在，跳过
        if node_key in node_mapping:
            nodes_skipped += 1
            continue
        
        # 添加节点到图中
        node_id = node_id_counter
        graph.add_node(node_id, pos=(x, y))
        node_mapping[node_key] = node_id
        node_id_counter += 1
        nodes_added += 1
    
    if pc_range is not None:
        print(f"\n[DEBUG graph_converter] Node processing summary:")
        print(f"  Total nodes in graph_nodelist: {len(eval_graph.graph_nodelist)}")
        print(f"  Nodes added to networkx: {nodes_added}")
        print(f"  Nodes skipped: {nodes_skipped}")
        print(f"  Final graph nodes: {len(graph.nodes())}")
    
    # 第二步：添加边（基于childs关系）
    for eval_node in eval_graph.graph_nodelist:
        if not hasattr(eval_node, 'coord') or not hasattr(eval_node, 'childs'):
            continue
        
        # 获取起始节点
        start_coord = eval_node.coord
        try:
            if not (hasattr(start_coord, '__len__') and len(start_coord) >= 2):
                continue
            x1, y1 = float(start_coord[0]), float(start_coord[1])
        except (TypeError, IndexError, ValueError):
            continue
        
        x1, y1 = transform_coord(x1, y1)
        
        start_key = get_node_key((x1, y1))
        if start_key not in node_mapping:
            continue
        
        start_id = node_mapping[start_key]
        
        # 遍历所有子节点
        for child_info in eval_node.childs:
            if isinstance(child_info, tuple) and len(child_info) >= 1:
                child_node = child_info[0]
                # bezier_coeff = child_info[1] if len(child_info) > 1 else None
            else:
                child_node = child_info
            
            if not hasattr(child_node, 'coord'):
                continue
            
            # 获取终止节点
            end_coord = child_node.coord
            try:
                if not (hasattr(end_coord, '__len__') and len(end_coord) >= 2):
                    continue
                x2, y2 = float(end_coord[0]), float(end_coord[1])
            except (TypeError, IndexError, ValueError):
                continue
            x2, y2 = transform_coord(x2, y2)
            
            end_key = get_node_key((x2, y2))
            if end_key not in node_mapping:
                continue
            
            end_id = node_mapping[end_key]
            
            # 添加边（避免重复）
            if not graph.has_edge(start_id, end_id):
                graph.add_edge(start_id, end_id)
    
    # 模式处理：插值模式（可选）
    if mode == 'interpolated' and interp_points > 2:
        graph = _add_interpolated_nodes(graph, interp_points)
    
    return graph


def _add_interpolated_nodes(graph, num_points):
    """
    在图的边上添加插值节点
    
    Args:
        graph: networkx.DiGraph
        num_points: 每条边的插值点数
    
    Returns:
        添加了插值节点的图
    """
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(graph.nodes(data=True))
    
    node_id_counter = max(graph.nodes()) + 1 if len(graph.nodes()) > 0 else 0
    
    for u, v in graph.edges():
        pos_u = graph.nodes[u]['pos']
        pos_v = graph.nodes[v]['pos']
        
        # 计算中间插值点
        prev_id = u
        for i in range(1, num_points):
            t = i / num_points
            x = pos_u[0] * (1 - t) + pos_v[0] * t
            y = pos_u[1] * (1 - t) + pos_v[1] * t
            
            # 添加插值节点
            new_id = node_id_counter
            new_graph.add_node(new_id, pos=(x, y))
            new_graph.add_edge(prev_id, new_id)
            
            prev_id = new_id
            node_id_counter += 1
        
        # 连接最后一个插值节点到终点
        new_graph.add_edge(prev_id, v)
    
    return new_graph


def eval_seq2graph_to_geotopo_format(eval_graph):
    """
    直接将EvalSeq2Graph转换为GEO/TOPO评价所需的字典格式
    这是一个更高效的转换方式（不经过networkx）
    
    Args:
        eval_graph: EvalSeq2Graph对象
    
    Returns:
        字典格式: {(x, y): [(x2, y2), ...]} 表示邻接关系
    """
    neighbors = {}
    
    if not hasattr(eval_graph, 'graph_nodelist'):
        return neighbors
    
    for eval_node in eval_graph.graph_nodelist:
        if not hasattr(eval_node, 'coord') or not hasattr(eval_node, 'childs'):
            continue
        
        # 获取节点坐标
        coord = eval_node.coord
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            continue
        
        x1, y1 = float(coord[0]), float(coord[1])
        k1 = (int(x1 * 10), int(y1 * 10))  # 放大10倍并取整，与CGNet保持一致
        
        if k1 not in neighbors:
            neighbors[k1] = []
        
        # 添加所有子节点
        for child_info in eval_node.childs:
            if isinstance(child_info, tuple) and len(child_info) >= 1:
                child_node = child_info[0]
            else:
                child_node = child_info
            
            if not hasattr(child_node, 'coord'):
                continue
            
            child_coord = child_node.coord
            if not isinstance(child_coord, (list, tuple)) or len(child_coord) < 2:
                continue
            
            x2, y2 = float(child_coord[0]), float(child_coord[1])
            k2 = (int(x2 * 10), int(y2 * 10))
            
            if k2 not in neighbors[k1]:
                neighbors[k1].append(k2)
    
    return neighbors
