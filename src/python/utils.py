import torch

def calculate_model_size(model):
    """Calculate the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time(model, input_tensor, num_runs=100):
    """Measure the average inference time of the model."""
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    # GPU warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # Measure performance
    with torch.no_grad():
        for _ in range(num_runs):
            starter.record()
            _ = model(input_tensor)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

    avg_time = sum(timings) / len(timings)
    return avg_time