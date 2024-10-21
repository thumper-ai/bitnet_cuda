import torch
import pytest
from bitnet import BitLinear, BitLayerNorm, BitNetModel

def test_bit_linear_cpu():
    layer = BitLinear(64, 32)
    input_tensor = torch.randn(16, 64)
    output = layer(input_tensor)
    assert output.shape == (16, 32)

def test_bit_layer_norm_cpu():
    norm = BitLayerNorm(64)
    input_tensor = torch.randn(16, 64)
    output = norm(input_tensor)
    assert output.shape == (16, 64)

def test_bitnet_model_cpu():
    model = BitNetModel(64, 128, 32, 4)
    input_tensor = torch.randn(16, 10, 64)
    output = model(input_tensor)
    assert output.shape == (16, 10, 32)