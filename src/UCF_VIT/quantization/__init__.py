"""
Custom Quantization Package for UCF-VIT
AMD GPU optimized implementation
"""

from .int8_quantization import (
    Int8Quantizer,
    QuantizedLinear,
    quantize_model_int8,
    replace_linear_with_quantized
)

from .extreme_quantization import (
    ExtremeQuantizer,
    BinaryQuantizer,
    QuantizedLinearExtreme,
    quantize_model_extreme,
    replace_linear_with_extreme_quantized,
    get_model_size_mb,
    get_quantized_model_size_mb
)

__all__ = [
    'Int8Quantizer',
    'QuantizedLinear', 
    'quantize_model_int8',
    'replace_linear_with_quantized',
    'ExtremeQuantizer',
    'BinaryQuantizer', 
    'QuantizedLinearExtreme',
    'quantize_model_extreme',
    'replace_linear_with_extreme_quantized',
    'get_model_size_mb',
    'get_quantized_model_size_mb'
]