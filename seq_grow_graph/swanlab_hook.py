# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Dict, Any
import os.path as osp

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class SwanLabLoggerHook(Hook):
    """SwanLab logger hook for mmengine.
    
    This hook logs training/validation metrics to SwanLab for visualization.
    
    Args:
        project (str): SwanLab project name. Default: 'SeqGrowGraph'
        experiment_name (str, optional): Experiment name. If None, will use config filename.
        log_interval (int): Logging interval. Default: 50
        log_model (bool): Whether to log model checkpoints. Default: False
        init_kwargs (dict, optional): Additional kwargs for swanlab.init()
    """
    
    priority = 'VERY_LOW'
    
    def __init__(self,
                 project: str = 'SeqGrowGraph',
                 experiment_name: Optional[str] = None,
                 log_interval: int = 50,
                 log_model: bool = False,
                 init_kwargs: Optional[Dict[str, Any]] = None):
        self.project = project
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        self.log_model = log_model
        self.init_kwargs = init_kwargs if init_kwargs is not None else {}
        self._swanlab = None
        
    def before_run(self, runner: Runner) -> None:
        """Initialize SwanLab before training starts."""
        # Only initialize SwanLab on rank 0 (main process) in distributed training
        if runner.rank != 0:
            return
        
        try:
            import swanlab
            self._swanlab = swanlab
        except ImportError:
            raise ImportError(
                'Please install swanlab with: pip install swanlab')
        
        # Get experiment name from config if not specified
        if self.experiment_name is None:
            cfg_filename = osp.basename(runner.cfg_file) if hasattr(runner, 'cfg_file') else 'experiment'
            self.experiment_name = osp.splitext(cfg_filename)[0]
        
        # Initialize SwanLab
        init_kwargs = dict(
            project=self.project,
            experiment_name=self.experiment_name,
            config=runner.cfg.to_dict() if hasattr(runner.cfg, 'to_dict') else dict(runner.cfg),
            logdir=runner.work_dir,
        )
        init_kwargs.update(self.init_kwargs)
        
        self._swanlab.init(**init_kwargs)
        runner.logger.info(f'SwanLab logging to project: {self.project}')
    
    def after_train_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[dict] = None) -> None:
        """Log training metrics after each iteration."""
        # Only log on rank 0
        if runner.rank != 0 or self._swanlab is None:
            return
            
        if self.every_n_inner_iters(batch_idx, self.log_interval):
            # Get current learning rate
            lr = runner.optim_wrapper.get_lr()
            if isinstance(lr, dict):
                for key, value in lr.items():
                    self._swanlab.log({f'train/lr_{key}': value[0] if isinstance(value, list) else value},
                                     step=runner.iter)
            else:
                lr_value = lr[0] if isinstance(lr, list) else lr
                self._swanlab.log({'train/lr': lr_value}, step=runner.iter)
            
            # Log training losses
            if 'log_vars' in runner.message_hub.log_scalars:
                log_vars = runner.message_hub.log_scalars['log_vars']
                
                train_metrics = {}
                for key, value in log_vars.items():
                    if isinstance(value, (int, float)):
                        # Separate loss and other metrics
                        if 'loss' in key.lower():
                            train_metrics[f'train/{key}'] = value
                        else:
                            train_metrics[f'train/{key}'] = value
                
                if train_metrics:
                    self._swanlab.log(train_metrics, step=runner.iter)
    
    def after_val_epoch(self,
                       runner: Runner,
                       metrics: Optional[Dict[str, float]] = None) -> None:
        """Log validation metrics after each validation epoch."""
        # Only log on rank 0
        if runner.rank != 0 or self._swanlab is None:
            return
            
        if metrics is not None:
            val_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Clean up metric names (remove prefixes if present)
                    clean_key = key.split('/')[-1] if '/' in key else key
                    val_metrics[f'val/{clean_key}'] = value
            
            if val_metrics:
                self._swanlab.log(val_metrics, step=runner.epoch)
                runner.logger.info(f'Logged {len(val_metrics)} validation metrics to SwanLab')
    
    def after_test_epoch(self,
                        runner: Runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Log test metrics after testing."""
        # Only log on rank 0
        if runner.rank != 0 or self._swanlab is None:
            return
            
        if metrics is not None:
            test_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    clean_key = key.split('/')[-1] if '/' in key else key
                    test_metrics[f'test/{clean_key}'] = value
            
            if test_metrics:
                self._swanlab.log(test_metrics)
                runner.logger.info(f'Logged {len(test_metrics)} test metrics to SwanLab')
    
    def after_run(self, runner: Runner) -> None:
        """Finalize SwanLab logging after training completes."""
        # Only finish on rank 0
        if runner.rank != 0 or self._swanlab is None:
            return
            
        self._swanlab.finish()
        runner.logger.info('SwanLab logging finished')
