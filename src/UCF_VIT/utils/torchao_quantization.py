"""
Torch.AO Quantization Module for UCF-VIT
Compatible with CatsDogs dataset and Frontier supercomputer
"""

import torch
import torch.nn as nn
from torch.ao.quantization import (
    quantize_dynamic,
    prepare_qat,
    convert,
    get_default_qconfig,
    QConfigMapping,
    default_dynamic_qconfig,
    default_qat_qconfig
)
# from torch.ao.quantization.qconfig import default_static_qconfig  # Not available in this PyTorch version
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize, default_fake_quant
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class TorchAOQuantizationConfig:
    """Configuration for torch.ao quantization"""
    
    def __init__(self, config_dict: Dict):
        self.enabled = config_dict.get('enabled', False)
        self.bits = config_dict.get('bits', 8)
        self.method = config_dict.get('method', 'qat')  # qat, static, dynamic
        self.quantize_weights = config_dict.get('quantize_weights', True)
        self.quantize_activations = config_dict.get('quantize_activations', True)
        self.calibration_samples = config_dict.get('calibration_samples', 100)
        self.quantize_layers = config_dict.get('quantize_layers', ['linear', 'conv'])
        self.exclude_layers = config_dict.get('exclude_layers', ['cls_token', 'pos_embed'])
        self.rocm_optimizations = config_dict.get('rocm_optimizations', False)
        self.performance_mode = config_dict.get('performance_mode', 'normal')
        self.profile_quantization = config_dict.get('profile_quantization', False)
        
    def get_qconfig(self) -> QConfig:
        """Get quantization configuration based on settings"""
        if self.bits == 8:
            if self.method == 'qat':
                return default_qat_qconfig
            else:  # dynamic (static not available)
                return default_dynamic_qconfig
        else:
            # For 4-bit, use custom qconfig
            return QConfig(
                activation=FakeQuantize.with_args(
                    observer=MinMaxObserver,
                    quant_min=0,
                    quant_max=15,  # 4-bit: 0-15
                    dtype=torch.quint8
                ),
                weight=FakeQuantize.with_args(
                    observer=MinMaxObserver,
                    quant_min=-8,
                    quant_max=7,  # 4-bit signed: -8 to 7
                    dtype=torch.qint8
                )
            )

class TorchAOQuantizer:
    """Torch.AO based quantizer for UCF-VIT models"""
    
    def __init__(self, config: TorchAOQuantizationConfig):
        self.config = config
        self.quantization_bits = config.bits
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply torch.ao quantization to model"""
        logger.info(f"Applying torch.ao {self.config.bits}-bit quantization...")
        
        try:
            if self.config.method == 'dynamic':
                return self._quantize_dynamic(model)
            elif self.config.method == 'static':
                # Static quantization not available, fallback to dynamic
                logger.warning("Static quantization not available, falling back to dynamic")
                return self._quantize_dynamic(model)
            elif self.config.method == 'qat':
                return self._quantize_qat(model)
            else:
                raise ValueError(f"Unsupported quantization method: {self.config.method}")
                
        except Exception as e:
            logger.error(f"Torch.ao quantization failed: {e}")
            raise
    
    def _quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization (weights only)"""
        logger.info("Applying dynamic quantization...")
        
        # Move model to CPU for quantization (ROCm doesn't support quantized ops)
        original_device = next(model.parameters()).device
        model_cpu = model.cpu()
        
        # Dynamic quantization only quantizes weights
        quantized_model = quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.Conv2d},  # Only quantize these layer types
            dtype=torch.qint8
        )
        
        # Move back to original device
        quantized_model = quantized_model.to(original_device)
        
        return quantized_model
    
    def _quantize_static(self, model: nn.Module) -> nn.Module:
        """Apply static quantization (weights + activations)"""
        logger.info("Applying static quantization...")
        
        # Set quantization config
        model.qconfig = self.config.get_qconfig()
        
        # Prepare model for quantization
        from torch.ao.quantization.quantize import prepare
        prepared_model = prepare(model)
        
        # Calibrate with dummy data
        self._calibrate_model(prepared_model)
        
        # Convert to quantized model
        quantized_model = convert(prepared_model)
        
        return quantized_model
    
    def _quantize_qat(self, model: nn.Module) -> nn.Module:
        """Apply Quantization Aware Training"""
        logger.info("Applying QAT quantization...")
        
        # Set quantization config
        model.qconfig = self.config.get_qconfig()
        
        # Prepare for QAT
        prepared_model = prepare_qat(model)
        
        # Note: In real training, you would train the prepared_model
        # For inference, we convert directly
        quantized_model = convert(prepared_model)
        
        return quantized_model
    
    def _calibrate_model(self, model: nn.Module, num_samples: int = 100):
        """Calibrate model with dummy data"""
        logger.info(f"Calibrating model with {num_samples} samples...")
        
        model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                # Create dummy input matching CatsDogs format
                dummy_input = torch.randn(1, 3, 256, 256)  # [batch, channels, height, width]
                dummy_variables = torch.randn(1, 3)  # [batch, variables]
                
                try:
                    # Forward pass for calibration
                    _ = model(dummy_input, dummy_variables, None)
                except:
                    # If model doesn't match expected signature, try simpler forward
                    _ = model(dummy_input)
    
    def get_quantization_stats(self, model: nn.Module) -> Dict:
        """Get quantization statistics"""
        stats = {
            'quantized_layers': 0,
            'total_layers': 0,
            'quantization_ratio': 0.0,
            'estimated_memory_reduction': 0.0,
            'quantization_bits': self.config.bits,
            'method': self.config.method
        }
        
        for name, module in model.named_modules():
            stats['total_layers'] += 1
            
            # Check if module is quantized
            if hasattr(module, 'weight_fake_quant') or hasattr(module, 'weight'):
                if hasattr(module, 'weight_fake_quant') or str(type(module)).find('Quantized') != -1:
                    stats['quantized_layers'] += 1
        
        if stats['total_layers'] > 0:
            stats['quantization_ratio'] = stats['quantized_layers'] / stats['total_layers']
        
        # Estimate memory reduction
        if self.config.bits == 8:
            stats['estimated_memory_reduction'] = 0.5
        elif self.config.bits == 4:
            stats['estimated_memory_reduction'] = 0.75
        elif self.config.bits == 2:
            stats['estimated_memory_reduction'] = 0.875
            
        return stats

def setup_torchao_quantization(model: nn.Module, config_dict: Dict) -> nn.Module:
    """Setup torch.ao quantization for model"""
    config = TorchAOQuantizationConfig(config_dict)
    
    if not config.enabled:
        logger.info("Torch.ao quantization disabled")
        return model
    
    # Check if ROCm environment (torch.ao quantized ops not supported)
    if torch.cuda.is_available() and hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_built():
        logger.warning("Torch.ao quantization not supported on ROCm/AMD GPU")
        logger.warning("Falling back to unquantized model")
        return model
    
    try:
        quantizer = TorchAOQuantizer(config)
        quantized_model = quantizer.quantize_model(model)
        
        # Log statistics
        stats = quantizer.get_quantization_stats(quantized_model)
        logger.info(f"Torch.ao quantization stats: {stats}")
        
        return quantized_model
    except Exception as e:
        logger.error(f"Torch.ao quantization failed: {e}")
        logger.warning("Falling back to unquantized model")
        return model

def add_torchao_quantization_args(parser):
    """Add torch.ao quantization command line arguments"""
    parser.add_argument('--torchao-quantization', action='store_true', default=False,
                       help='Enable torch.ao quantization')
    parser.add_argument('--torchao-bits', type=int, choices=[4, 8], default=8,
                       help='Torch.ao quantization bits')
    parser.add_argument('--torchao-method', type=str, 
                       choices=['dynamic', 'static', 'qat'], default='dynamic',
                       help='Torch.ao quantization method')
    parser.add_argument('--torchao-weights', action='store_true', default=True,
                       help='Quantize model weights')
    parser.add_argument('--torchao-activations', action='store_true', default=False,
                       help='Quantize model activations')
    return parser

def create_torchao_config_from_args(args) -> Dict:
    """Create torch.ao quantization config from command line arguments"""
    return {
        'enabled': getattr(args, 'torchao_quantization', False),
        'bits': getattr(args, 'torchao_bits', 8),
        'method': getattr(args, 'torchao_method', 'dynamic'),
        'quantize_weights': getattr(args, 'torchao_weights', True),
        'quantize_activations': getattr(args, 'torchao_activations', False),
        'calibration_samples': 100,
        'quantize_layers': ['linear', 'conv'],
        'exclude_layers': ['cls_token', 'pos_embed'],
        'rocm_optimizations': False,
        'performance_mode': 'normal',
        'profile_quantization': False
    }
