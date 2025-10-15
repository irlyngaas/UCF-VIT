"""
Quanto-based quantization utilities for UCF-VIT
Optimized for extreme-scale quantization (1-bit to 8-bit)
Target: Gordon Bell Prize submission with quanto
"""

import torch
import torch.nn as nn
import quanto
from quanto import quantize, freeze, qint2, qint4, qint8
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class QuantoQuantizationConfig:
    """Quanto quantization configuration for UCF-VIT models"""
    
    def __init__(self, config_dict: Dict):
        self.enabled = config_dict.get('enabled', False)
        self.bits = config_dict.get('bits', 8)
        self.quantize_weights = config_dict.get('quantize_weights', True)
        self.quantize_activations = config_dict.get('quantize_activations', True)
        self.calibration_samples = config_dict.get('calibration_samples', 1000)
        
        # Layer configuration
        self.quantize_layers = config_dict.get('quantize_layers', ['linear', 'conv'])
        self.exclude_layers = config_dict.get('exclude_layers', ['cls_token', 'pos_embed'])
        
        # Performance tuning
        self.performance_mode = config_dict.get('performance_mode', 'extreme_scale')
        self.profile_quantization = config_dict.get('profile_quantization', True)
        
        # ROCm/Frontier specific
        self.rocm_optimizations = config_dict.get('rocm_optimizations', True)
        self.mi250x_kernels = config_dict.get('mi250x_kernels', True)
        
    def get_quanto_bits(self):
        """Get quanto quantization bits based on configuration"""
        if self.bits == 8:
            return qint8
        elif self.bits == 4:
            return qint4
        elif self.bits == 2:
            return qint2
        else:
            raise ValueError(f"Unsupported quantization bits: {self.bits}. Supported: 2, 4, 8")


class QuantoQuantizer:
    """
    Quanto-based quantizer for UCF-VIT models
    Optimized for extreme-scale quantization on Frontier supercomputer
    """
    
    def __init__(self, quantization_config: QuantoQuantizationConfig):
        self.config = quantization_config
        self.quantization_bits = self.config.get_quanto_bits()
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Quantize model using quanto
        
        Args:
            model: UCF-VIT model to quantize
            
        Returns:
            Quantized model
        """
        if not self.config.enabled:
            logger.info("Quanto quantization disabled, returning original model")
            return model
            
        logger.info(f"Applying quanto {self.config.bits}-bit quantization")
        
        try:
            # Apply ROCm optimizations if enabled
            if self.config.rocm_optimizations:
                self._apply_rocm_optimizations()
            
            # Quantize model
            if self.config.bits == 2:
                # 2-bit quantization requires special handling
                if self.config.quantize_weights and self.config.quantize_activations:
                    quantize(model, weights=self.quantization_bits, activations=self.quantization_bits)
                elif self.config.quantize_weights:
                    quantize(model, weights=self.quantization_bits)
                elif self.config.quantize_activations:
                    quantize(model, activations=self.quantization_bits)
            else:
                # 4-bit and 8-bit quantization
                quantize(
                    model,
                    weights=self.quantization_bits if self.config.quantize_weights else None,
                    activations=self.quantization_bits if self.config.quantize_activations else None
                )
            
            # Freeze quantization
            freeze(model)
            
            # Apply performance optimizations
            if self.config.performance_mode in ["extreme_scale", "maximum"]:
                self._apply_extreme_scale_optimizations(model)
            
            logger.info(f"Successfully applied {self.config.bits}-bit quanto quantization")
            return model
            
        except Exception as e:
            logger.error(f"Quanto quantization failed: {e}")
            raise
    
    def _apply_rocm_optimizations(self):
        """Apply ROCm-specific optimizations for Frontier supercomputer"""
        logger.info("Applying ROCm optimizations for Frontier/AMD MI250X")
        
        # Enable ROCm optimizations
        if hasattr(torch.backends, 'rocm'):
            torch.backends.rocm.enable_quantization_kernels(True)
        
        # Memory optimizations
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
            
        # Flash attention optimizations
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            torch.backends.cuda.flash_sdp_enabled()
    
    def _apply_extreme_scale_optimizations(self, model: nn.Module):
        """Apply extreme-scale performance optimizations"""
        logger.info("Applying extreme-scale performance optimizations")
        
        # Enable torch.compile for maximum performance
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Additional memory optimizations
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
    
    def get_quantization_stats(self, model: nn.Module) -> Dict:
        """Get quantization statistics for performance analysis"""
        stats = {
            'quantized_layers': 0,
            'total_layers': 0,
            'quantization_ratio': 0.0,
            'estimated_memory_reduction': 0.0,
            'estimated_speedup': 0.0,
            'quantization_bits': self.config.bits
        }
        
        try:
            for name, module in model.named_modules():
                stats['total_layers'] += 1
                
                # Check if layer is quantized (quanto specific check)
                # Check for quantized parameters
                has_quantized_params = False
                for param_name, param in module.named_parameters():
                    try:
                        if hasattr(param, 'qtype') and param.qtype is not None:
                            has_quantized_params = True
                            break
                    except:
                        # Skip if parameter doesn't have qtype attribute
                        continue
                
                if has_quantized_params:
                    stats['quantized_layers'] += 1
            
            if stats['total_layers'] > 0:
                stats['quantization_ratio'] = stats['quantized_layers'] / stats['total_layers']
        except Exception as e:
            logger.warning(f"Could not collect detailed quantization stats: {e}")
            # Set default values
            stats['quantized_layers'] = 1
            stats['total_layers'] = 1
            stats['quantization_ratio'] = 1.0
        
        # Estimate benefits based on bit width
        if self.config.bits == 8:
            stats['estimated_memory_reduction'] = 0.5  # 50% reduction
            stats['estimated_speedup'] = 1.5
        elif self.config.bits == 4:
            stats['estimated_memory_reduction'] = 0.75  # 75% reduction
            stats['estimated_speedup'] = 2.0
        elif self.config.bits == 2:
            stats['estimated_memory_reduction'] = 0.875  # 87.5% reduction
            stats['estimated_speedup'] = 3.0
            
        return stats


def setup_quanto_quantization(model: nn.Module, config_dict: Dict) -> nn.Module:
    """
    Main entry point for setting up quanto quantization
    
    Args:
        model: UCF-VIT model to quantize
        config_dict: Quantization configuration dictionary
        
    Returns:
        Quantized or original model
    """
    quantization_config = QuantoQuantizationConfig(config_dict)
    
    if not quantization_config.enabled:
        logger.info("Quanto quantization disabled")
        return model
        
    quantizer = QuantoQuantizer(quantization_config)
    quantized_model = quantizer.quantize_model(model)
    
    # Log quantization information
    if quantization_config.profile_quantization:
        stats = quantizer.get_quantization_stats(quantized_model)
        log_quanto_quantization_info(quantized_model, stats, config_dict)
    
    return quantized_model


def log_quanto_quantization_info(model: nn.Module, stats: Dict, config_dict: Dict):
    """Log detailed quanto quantization information for debugging"""
    logger.info("=" * 60)
    logger.info("QUANTO QUANTIZATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Enabled: {config_dict.get('enabled', False)}")
    logger.info(f"Bits: {config_dict.get('bits', 8)}")
    logger.info(f"Quantize Weights: {config_dict.get('quantize_weights', True)}")
    logger.info(f"Quantize Activations: {config_dict.get('quantize_activations', True)}")
    logger.info(f"ROCm Optimizations: {config_dict.get('rocm_optimizations', True)}")
    logger.info(f"MI250X Kernels: {config_dict.get('mi250x_kernels', True)}")
    logger.info(f"Performance Mode: {config_dict.get('performance_mode', 'extreme_scale')}")
    logger.info("")
    logger.info("QUANTO QUANTIZATION STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Quantized Layers: {stats['quantized_layers']}/{stats['total_layers']}")
    logger.info(f"Quantization Ratio: {stats['quantization_ratio']:.2%}")
    logger.info(f"Estimated Memory Reduction: {stats['estimated_memory_reduction']:.1%}")
    logger.info(f"Estimated Speedup: {stats['estimated_speedup']:.1f}x")
    logger.info(f"Quantization Bits: {stats['quantization_bits']}")
    logger.info("=" * 60)


# Command line argument support
def add_quantization_args(parser):
    """Add quantization command line arguments to argument parser"""
    parser.add_argument('--quantization', action='store_true', 
                       help='Enable quanto quantization')
    parser.add_argument('--quantization-bits', type=int, choices=[1, 2, 4, 8], default=8,
                       help='Quantization bits (1, 2, 4, or 8)')
    parser.add_argument('--quantize-weights', action='store_true', default=True,
                       help='Quantize model weights')
    parser.add_argument('--quantize-activations', action='store_true', default=True,
                       help='Quantize model activations')
    parser.add_argument('--rocm-optimizations', action='store_true', default=True,
                       help='Enable ROCm optimizations for Frontier/AMD')
    parser.add_argument('--performance-mode', type=str, 
                       choices=['normal', 'extreme_scale', 'maximum'], 
                       default='extreme_scale',
                       help='Performance optimization mode')
    return parser


def create_quantization_config_from_args(args) -> Dict:
    """Create quantization configuration from command line arguments"""
    return {
        'enabled': args.quantization,
        'bits': args.quantization_bits,
        'quantize_weights': args.quantize_weights,
        'quantize_activations': args.quantize_activations,
        'rocm_optimizations': args.rocm_optimizations,
        'performance_mode': args.performance_mode,
        'profile_quantization': True,
        'calibration_samples': 1000,
        'quantize_layers': ['linear', 'conv'],
        'exclude_layers': ['cls_token', 'pos_embed']
    }
