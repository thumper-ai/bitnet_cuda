# pytorch_integration.py
import torch
from torch.utils.cpp_extension import load

bitnet_cuda = load(name="bitnet_cuda", sources=["bitnet_cuda.cpp", "bitnet_cuda_pytorch.cu"])

class BitNetLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.bitnet = bitnet_cuda.BitNetCUDA(hidden_size, num_layers)

    def forward(self, x):
        return self.bitnet.forward(x)

# tensorflow_integration.py
import tensorflow as tf
from tensorflow.python.framework import ops

bitnet_cuda_module = tf.load_op_library('./libbitnet_cuda.so')

@ops.RegisterGradient("BitNetForward")
def _bitnet_backward(op, grad):
    return bitnet_cuda_module.bitnet_backward(grad, op.inputs[0], op.inputs[1])

def bitnet_layer(inputs, hidden_size, num_layers):
    return bitnet_cuda_module.bitnet_forward(inputs, hidden_size, num_layers)