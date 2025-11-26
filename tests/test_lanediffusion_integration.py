"""
简单测试脚本 - 验证LaneDiffusion指标集成是否正常工作
"""

import sys
from pathlib import Path
import networkx as nx

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """测试所有模块能否正常导入"""
    print("测试1: 检查模块导入...")
    try:
        from seq_grow_graph.evaluators.lanediffusion import GraphEvaluator
        from seq_grow_graph.evaluators import HybridEvaluator
        from seq_grow_graph.adapters import eval_seq2graph_to_networkx
        print("✓ 所有模块导入成功!")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_graph_evaluator():
    """测试GraphEvaluator基本功能"""
    print("\n测试2: GraphEvaluator基本功能...")
    try:
        from seq_grow_graph.evaluators.lanediffusion import GraphEvaluator
        
        # 创建简单测试图
        gt_graph = nx.DiGraph()
        gt_graph.add_node(0, pos=(0, 0))
        gt_graph.add_node(1, pos=(10, 0))
        gt_graph.add_edge(0, 1)
        
        pred_graph = nx.DiGraph()
        pred_graph.add_node(0, pos=(0, 0))
        pred_graph.add_node(1, pos=(10, 0))
        pred_graph.add_edge(0, 1)
        
        # 评价
        evaluator = GraphEvaluator(radius=5)
        metrics = evaluator.evaluate_graph(gt_graph, pred_graph)
        
        # 检查指标
        assert 'GEO F1' in metrics
        assert 'TOPO F1' in metrics
        assert 'Graph IoU' in metrics
        assert metrics['GEO F1'] > 0  # 完全匹配应该有很高的分数
        
        print(f"✓ GraphEvaluator工作正常!")
        print(f"  GEO F1: {metrics['GEO F1']:.3f}")
        print(f"  TOPO F1: {metrics['TOPO F1']:.3f}")
        print(f"  IoU: {metrics['Graph IoU']:.3f}")
        return True
    except Exception as e:
        print(f"✗ GraphEvaluator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_evaluator():
    """测试HybridEvaluator"""
    print("\n测试3: HybridEvaluator...")
    try:
        from seq_grow_graph.evaluators import HybridEvaluator
        
        # 初始化
        evaluator = HybridEvaluator(
            lanediff_radius=5,
            conversion_mode='simple'
        )
        
        # 创建dummy metrics测试格式化输出
        dummy_metrics = {
            'LaneDiff/GEO F1': 0.85,
            'LaneDiff/TOPO F1': 0.72,
            'LaneDiff/Graph IoU': 0.68
        }
        
        result_str = evaluator.format_results(dummy_metrics)
        assert len(result_str) > 0
        
        print("✓ HybridEvaluator工作正常!")
        return True
    except Exception as e:
        print(f"✗ HybridEvaluator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_converter():
    """测试图转换功能"""
    print("\n测试4: 图转换功能...")
    try:
        from seq_grow_graph.adapters import get_node_key
        
        # 测试节点键生成
        key1 = get_node_key((10.123, 20.456))
        key2 = get_node_key((10.126, 20.458))  # 接近的坐标
        
        # 应该映射到相同的键（因为round到2位小数）
        assert key1 == key2
        
        print("✓ 图转换功能正常!")
        return True
    except Exception as e:
        print(f"✗ 图转换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("="  * 60)
    print("LaneDiffusion集成测试")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_graph_evaluator,
        test_hybrid_evaluator,
        test_graph_converter
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过: {sum(results)}/{len(results)}")
    print(f"失败: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ 所有测试通过! LaneDiffusion指标集成成功!")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查错误信息")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
