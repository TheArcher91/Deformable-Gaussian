"""
Deformable Gaussian Splatting Config
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

# Splatfacto specific imports
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

# Import your custom model configuration
from method_template.deformable_model import DeformableSplatfactoModelConfig

method_template = MethodSpecification(
    config=TrainerConfig(
        method_name="deformable-splat",
        
        # 1. DISABLE MID-TRAINING EVALUATIONS
        # This prevents the massive VRAM spikes that cause crashes
        steps_per_eval_image=0,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=0,
        
        # 2. REDUCE CHECKPOINT FREQUENCY
        # Only save exactly halfway and at the end
        steps_per_save=15000, 
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=DNeRFDataParserConfig(scale_factor=0.5), 
                cache_images="cpu", 
            ),
            model=DeformableSplatfactoModelConfig(
                # 3. GEOMETRY & MATH OPTIMIZATIONS
                sh_degree=2,            # Less color math per point = much faster ETA
                cull_alpha_thresh=0.01  # Delete ghost/transparent points more aggressively
            ),
        ),
        
        # ... [KEEP YOUR EXISTING OPTIMIZERS DICTIONARY HERE] ...
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
            # YOUR CUSTOM MLP OPTIMIZER
            "deformation_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=30000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Deformable 3D Gaussian Splatting architecture for dynamic scenes.",
)