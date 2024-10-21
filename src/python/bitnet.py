import torch
import bitnet_cuda

class BitLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_scale = torch.nn.Parameter(torch.Tensor(1))
        self.input_scale = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_scale.data.fill_(1.0)
        self.input_scale.data.fill_(1.0)

    def forward(self, input):
        quantized_weight = bitnet_cuda.quantize_weights(self.weight, self.weight_scale)
        return bitnet_cuda.fused_quantize_matmul(input, quantized_weight, self.input_scale, self.weight_scale)

class BitLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(BitLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = torch.nn.Parameter(torch.Tensor(normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return bitnet_cuda.layernorm(input, self.weight, self.bias, self.eps)

class BitNetModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BitNetModel, self).__init__()
        self.layers = torch.nn.ModuleList([
            BitLinear(input_dim if i == 0 else hidden_dim, hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([
            BitLayerNorm(hidden_dim)
            for _ in range(num_layers - 1)
        ])

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x = norm(torch.nn.functional.relu(layer(x)))
        return self.layers[-1](x)

# Usage example
if __name__ == "__main__":
    input_dim = 768
    hidden_dim = 3072
    output_dim = 768
    num_layers = 12
    batch_size = 32
    seq_length = 512

    model = BitNetModel(input_dim, hidden_dim, output_dim, num_layers).cuda()
    input_tensor = torch.randn(batch_size, seq_length, input_dim).cuda()

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")