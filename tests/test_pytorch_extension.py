import pytest
import torch
import numpy as np
from torch.utils.cpp_extension import load

# Load the custom CUDA extension
bitnet_cuda = load(name="bitnet_cuda", sources=["bitnet_cuda.cpp", "bitnet_cuda_kernel.cu"])

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def bitnet_model(device):
    # Create a simple BitNet model for testing
    class BitNetModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(BitNetModel, self).__init__()
            self.fc1 = bitnet_cuda.BitLinear(input_size, hidden_size)
            self.fc2 = bitnet_cuda.BitLinear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    return BitNetModel(784, 256, 10).to(device)

def test_bitnet_quantize_weights(device):
    weights = torch.randn(1024, 1024, device=device)
    quantized_weights, scale = bitnet_cuda.quantize_weights(weights)
    
    assert quantized_weights.shape == (1024, 64)  # 1024 x 1024 / 16 (packed)
    assert scale.shape == (1,)
    assert quantized_weights.dtype == torch.int32
    assert scale.dtype == torch.float32

    # Check if the quantized values are within the expected range (-1, 0, 1)
    unpacked = bitnet_cuda.unpack_ternary(quantized_weights)
    assert torch.all(torch.isin(unpacked, torch.tensor([-1, 0, 1], device=device)))

def test_bitnet_dequantize_weights(device):
    original_weights = torch.randn(1024, 1024, device=device)
    quantized_weights, scale = bitnet_cuda.quantize_weights(original_weights)
    dequantized_weights = bitnet_cuda.dequantize_weights(quantized_weights, scale, original_weights.shape)

    assert dequantized_weights.shape == original_weights.shape
    assert torch.allclose(original_weights, dequantized_weights, rtol=1e-2, atol=1e-2)

def test_bitnet_matmul(device):
    A = torch.randn(128, 256, device=device)
    B = torch.randn(256, 512, device=device)
    
    quantized_A, scale_A = bitnet_cuda.quantize_weights(A)
    quantized_B, scale_B = bitnet_cuda.quantize_weights(B)

    C_bitnet = bitnet_cuda.matmul(quantized_A, quantized_B, scale_A, scale_B)
    C_torch = torch.matmul(A, B)

    assert C_bitnet.shape == C_torch.shape
    assert torch.allclose(C_bitnet, C_torch, rtol=1e-2, atol=1e-2)

def test_bitnet_linear_layer(bitnet_model, device):
    input_data = torch.randn(32, 784, device=device)
    output = bitnet_model(input_data)

    assert output.shape == (32, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_bitnet_backward_pass(bitnet_model, device):
    input_data = torch.randn(32, 784, device=device, requires_grad=True)
    target = torch.randint(0, 10, (32,), device=device)
    
    optimizer = torch.optim.Adam(bitnet_model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Forward pass
    output = bitnet_model(input_data)
    loss = criterion(output, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if gradients are computed and not NaN
    for param in bitnet_model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_bitnet_performance(bitnet_model, device, batch_size):
    input_data = torch.randn(batch_size, 784, device=device)

    # Warm-up
    for _ in range(10):
        _ = bitnet_model(input_data)

    # Measure performance
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(100):
        _ = bitnet_model(input_data)
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 100  # Average time per forward pass

    print(f"Average forward pass time for batch size {batch_size}: {elapsed_time:.4f} ms")
    assert elapsed_time > 0

def test_bitnet_multi_gpu(device):
    if torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 GPUs")

    model = torch.nn.DataParallel(bitnet_model(device))
    input_data = torch.randn(64, 784, device=device)
    output = model(input_data)

    assert output.shape == (64, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

if __name__ == "__main__":
    pytest.main([__file__])

#     To run these tests, you'll need to have pytest installed (pip install pytest) and ensure that your CUDA extension is properly compiled and accessible. Run the tests using:
# text
# pytest tests/test_pytorch_extension.py

# This test suite will help ensure that your BitNet CUDA implementation integrates correctly with PyTorch and performs as expected. Remember to adjust the file paths and import statements as necessary to match your project structure.