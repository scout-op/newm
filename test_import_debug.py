#!/usr/bin/env python
"""
诊断脚本：测试 HybridEvaluator 导入链
"""
import sys
import os

print("=" * 80)
print("开始诊断 HybridEvaluator 导入问题")
print("=" * 80)

# 打印当前路径信息
print(f"\n当前工作目录: {os.getcwd()}")
print(f"\nPython 路径 (sys.path 前5项):")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

# 测试1: 检查依赖包
print("\n" + "=" * 80)
print("测试1: 检查依赖包")
print("=" * 80)

dependencies = ['numpy', 'networkx', 'rtree', 'shapely', 'scipy', 'cv2']
for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep} - OK")
    except ImportError as e:
        print(f"✗ {dep} - FAILED: {e}")

# 测试2: 检查包结构
print("\n" + "=" * 80)
print("测试2: 检查包结构")
print("=" * 80)

required_files = [
    'projects/SeqGrowGraph/seq_grow_graph/__init__.py',
    'projects/SeqGrowGraph/seq_grow_graph/evaluators/__init__.py',
    'projects/SeqGrowGraph/seq_grow_graph/evaluators/hybrid_evaluator.py',
    'projects/SeqGrowGraph/seq_grow_graph/evaluators/lanediffusion/__init__.py',
    'projects/SeqGrowGraph/seq_grow_graph/adapters/__init__.py',
    'projects/SeqGrowGraph/seq_grow_graph/adapters/graph_converter.py',
]

for filepath in required_files:
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
    else:
        print(f"✗ {filepath} - NOT FOUND")

# 测试3: 逐步导入测试
print("\n" + "=" * 80)
print("测试3: 逐步导入测试")
print("=" * 80)

# 3.1 导入 adapters
print("\n3.1 导入 graph_converter...")
try:
    from projects.SeqGrowGraph.seq_grow_graph.adapters import graph_converter
    print("✓ graph_converter 导入成功")
except Exception as e:
    print(f"✗ graph_converter 导入失败:")
    import traceback
    traceback.print_exc()

# 3.2 导入 lanediffusion
print("\n3.2 导入 lanediffusion...")
try:
    from projects.SeqGrowGraph.seq_grow_graph.evaluators import lanediffusion
    print("✓ lanediffusion 导入成功")
except Exception as e:
    print(f"✗ lanediffusion 导入失败:")
    import traceback
    traceback.print_exc()

# 3.3 导入 GraphEvaluator
print("\n3.3 导入 GraphEvaluator...")
try:
    from projects.SeqGrowGraph.seq_grow_graph.evaluators.lanediffusion import GraphEvaluator
    print("✓ GraphEvaluator 导入成功")
except Exception as e:
    print(f"✗ GraphEvaluator 导入失败:")
    import traceback
    traceback.print_exc()

# 3.4 导入 hybrid_evaluator
print("\n3.4 导入 hybrid_evaluator 模块...")
try:
    from projects.SeqGrowGraph.seq_grow_graph.evaluators import hybrid_evaluator
    print("✓ hybrid_evaluator 模块导入成功")
except Exception as e:
    print(f"✗ hybrid_evaluator 模块导入失败:")
    import traceback
    traceback.print_exc()

# 3.5 导入 HybridEvaluator 类
print("\n3.5 导入 HybridEvaluator 类...")
try:
    from projects.SeqGrowGraph.seq_grow_graph.evaluators import HybridEvaluator
    print("✓ HybridEvaluator 类导入成功")
    print(f"  HybridEvaluator: {HybridEvaluator}")
except Exception as e:
    print(f"✗ HybridEvaluator 类导入失败:")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
