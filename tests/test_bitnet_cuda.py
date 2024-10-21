import torch
import pytest
from bitnet import BitLinear, BitLayerNorm, BitNetModel

@pytest.mark.cuda
def test_bit_linear():
    layer = BitLinear(64, 32).cuda()
    input_tensor = torch.randn(16, 64).cuda()
    output = layer(input_tensor)
    assert output.shape == (16, 32)

@pytest.mark.cuda
def test_bit_layer_norm():
    norm = BitLayerNorm(64).cuda()
    input_tensor = torch.randn(16, 64).cuda()
    output = norm(input_tensor)
    assert output.shape == (16, 64)

@pytest.mark.cuda
def test_bitnet_model():
    model = BitNetModel(64, 128, 32, 4).cuda()
    input_tensor = torch.randn(16, 10, 64).cuda()
    output = model(input_tensor)
    assert output.shape == (16, 10, 32)