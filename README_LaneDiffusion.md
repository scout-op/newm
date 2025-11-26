# LaneDiffusion指标集成 - 使用文档

## 概述

本集成将LaneDiffusion论文的9个评价指标添加到SeqGrowGraph项目中，包括：

- GEO F1, TOPO F1, JTOPO F1（几何和拓扑F1分数）
- IoU（交并比）
- APLS（平均路径长度相似度）
- SDA（分叉点检测准确度）

## 安装依赖

```bash
cd /home/subobo/ro/newm/SeqGrowGraph
pip install -r requirements_lanediffusion.txt
```

## 快速测试

运行测试脚本验证集成是否成功：

```bash
python tests/test_lanediffusion_integration.py
```

预期输出：
```
====================================
LaneDiffusion集成测试
====================================
测试1: 检查模块导入...
✓ 所有模块导入成功!

测试2: GraphEvaluator基本功能...
✓ GraphEvaluator工作正常!
  GEO F1: 0.xxx
  TOPO F1: 0.xxx
  IoU: 0.xxx

...

✓ 所有测试通过! LaneDiffusion指标集成成功!
```

## 使用方法

### 方法1: 直接使用GraphEvaluator

```python
import networkx as nx
from seq_grow_graph.evaluators.lanediffusion import GraphEvaluator

# 创建或转换你的图为networkx.DiGraph格式
gt_graph = nx.DiGraph()
gt_graph.add_node(0, pos=(0, 0))
gt_graph.add_node(1, pos=(10, 0))
gt_graph.add_edge(0, 1)

pred_graph = ...  # 你的预测图

# 评价
evaluator = GraphEvaluator(radius=5)
metrics = evaluator.evaluate_graph(gt_graph, pred_graph)

print(f"GEO F1: {metrics['GEO F1']:.3f}")
print(f"TOPO F1: {metrics['TOPO F1']:.3f}")
```

### 方法2: 使用HybridEvaluator

```python
from seq_grow_graph.evaluators import HybridEvaluator

# 初始化
evaluator = HybridEvaluator(
    lanediff_radius=5,
    lanediff_interp_dist=2.5,
    lanediff_prop_dist=80
)

# 对单个样本评价
metrics = evaluator.evaluate_lanediffusion_single(
    gt_eval_graph,  # EvalSeq2Graph对象
    pred_eval_graph,
    verbose=True
)

# 格式化输出
print(evaluator.format_results(metrics))
```

### 方法3: 批量评价

```python
metrics = evaluator.evaluate_lanediffusion_batch(
    gt_graphs_list,
    pred_graphs_list,
    verbose=True
)
```

## 配置参数说明

### GraphEvaluator参数

- `radius` (default=8): GEO/TOPO匹配半径（米）
- `interp_dist` (default=2): 插值距离（米）
- `prop_dist` (default=400): 传播距离（米）
- `area_size` (default=[30, 60]): 渲染区域大小 [宽, 高]
- `lane_width` (default=1): 渲染车道宽度
- `gmode` (default='direct'): 图模式

### HybridEvaluator参数

- `conversion_mode`: 'simple'（只转换节点和边）或 'interpolated'（插值）
- `interp_points`: 插值点数量（mode='interpolated'时使用）

## 输出指标说明

| 指标 | 含义 | 范围 |
|------|------|------|
| GEO Precision/Recall/F1 | 几何位置匹配精度/召回率/F1 | [0, 1] |
| TOPO Precision/Recall/F1 | 拓扑连接精度/召回率/F1 | [0, 1] |
| JTOPO F1 | 交叉口拓扑F1分数 | [0, 1] |
| Graph IoU | 图的交并比 | [0, 1] |
| APLS | 平均路径长度相似度 | [0, 1] |
| SDA | 分叉点检测准确度 | [0, 1] |

## 常见问题

### Q1: 导入错误 "No module named 'rtree'"

```bash
pip install rtree
```

### Q2: 如何调整匹配阈值？

```python
evaluator = GraphEvaluator(radius=10)  # 增加匹配半径
```

### Q3: 评价很慢怎么办？

- 使用`conversion_mode='simple'`避免插值
- 减少`prop_dist`参数
- 使用批量评价并启用多进程

### Q4: 指标全为0是什么原因？

检查：
1. 图是否为空
2. 节点是否有`pos`属性
3. 坐标范围是否正确

## 示例

完整示例请参考：
```
examples/lanediffusion_example.py
```

## 技术支持

如有问题，请查看：
- 技术分析文档：`/home/subobo/ro/newm/LaneDiffusion指标集成分析.md`
- 实施计划：`implementation_plan.md`
