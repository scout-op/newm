#!/usr/bin/env python
"""
直接从 JSON 文件评价 SeqGrowGraph 和 LaneDiffusion 指标
无需重新推理，节省时间

使用方法:
python tools/eval_from_json.py \
    --json-path work_dirs/seq_grow_graph/results_nusc.json \
    --data-root ./data/nuscenes/ \
    --val-pkl ./data/nuscenes/nuscenes_centerline_infos_val.pkl \
    --config configs/seq_grow_graph/seq_grow_graph_default.py \
    --enable-lanediffusion
"""

import argparse
import pickle
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from projects.SeqGrowGraph.seq_grow_graph.transforms import BzRoadnetReachDistEvalNew
from mmengine import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate from JSON results')
    parser.add_argument('--json-path', required=True, help='Path to results JSON file')
    parser.add_argument('--data-root', default='./data/nuscenes/', help='Data root directory')
    parser.add_argument('--val-pkl', default=None, help='Path to validation pickle file')
    parser.add_argument('--config', default=None, help='Config file (optional, for reading grid_conf)')
    parser.add_argument('--enable-lanediffusion', action='store_true', help='Enable LaneDiffusion metrics')
    parser.add_argument('--num-proc', type=int, default=20, help='Number of processes for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 读取配置
    if args.config:
        cfg = Config.fromfile(args.config)
        grid_conf = cfg.grid_conf
        bz_grid_conf = cfg.bz_grid_conf
    else:
        # 使用默认配置
        grid_conf = dict(
            xbound=[-48.0, 48.0, 0.5],
            ybound=[-32.0, 32.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 48.0, 1.0],
        )
        bz_grid_conf = dict(
            xbound=[-55.0, 55.0, 0.5],
            ybound=[-55.0, 55.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 48.0, 1.0],
        )
    
    # 读取验证集信息
    if args.val_pkl is None:
        args.val_pkl = os.path.join(args.data_root, 'nuscenes_centerline_infos_val.pkl')
    
    logger.info(f"Loading validation data from: {args.val_pkl}")
    with open(args.val_pkl, 'rb') as f:
        data = pickle.load(f)
        # 处理不同的pickle格式
        if isinstance(data, dict):
            val_pkl = data.get('data_list', data.get('infos', list(data.values())[0] if data else []))
        elif isinstance(data, list):
            val_pkl = data
        else:
            raise ValueError(f"Unexpected pickle format: {type(data)}")
    
    logger.info(f"Total validation samples: {len(val_pkl)}")
    logger.info(f"Reading results from: {args.json_path}")
    
    # LaneDiffusion 配置
    lanediff_config = None
    if args.enable_lanediffusion:
        lanediff_config = dict(
            lanediff_radius=5,
            lanediff_interp_dist=2.5,
            lanediff_prop_dist=80,
            lanediff_area_size=[96, 64],
            lanediff_lane_width=2,
            conversion_mode='simple',
            interp_points=20
        )
        logger.info("LaneDiffusion metrics enabled")
    
    # 运行评价
    logger.info("Starting evaluation...")
    results = BzRoadnetReachDistEvalNew(
        result_path=args.json_path,
        data_root=args.data_root,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf,
        num_proc=args.num_proc,
        val_pkl=val_pkl,
        logger=logger,
        enable_lanediffusion=args.enable_lanediffusion,
        lanediff_config=lanediff_config
    )
    
    # 打印结果摘要
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 80)
    logger.info(f"Landmark F1 (mean): {results['landmark_f_score'].mean():.4f}")
    logger.info(f"Reachability F1 (mean): {results['reach_f_score'].mean():.4f}")
    
    if 'lanediff_metrics' in results and results['lanediff_metrics']:
        logger.info("\nLaneDiffusion Metrics:")
        for key, value in sorted(results['lanediff_metrics'].items()):
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info("=" * 80)
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
