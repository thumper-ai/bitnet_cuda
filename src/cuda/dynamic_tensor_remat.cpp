class DynamicTensorRematerialization {
private:
    struct TensorInfo {
        torch::Tensor tensor;
        std::function<torch::Tensor()> recompute_func;
        size_t size;
        int64_t last_use;
    };

    std::unordered_map<uintptr_t, TensorInfo> tensor_cache;
    size_t memory_limit;
    size_t current_memory_usage;
    int64_t step_counter;

public:
    DynamicTensorRematerialization(size_t memory_limit) 
        : memory_limit(memory_limit), current_memory_usage(0), step_counter(0) {}

    torch::Tensor get_tensor(uintptr_t key, std::function<torch::Tensor()> recompute_func) {
        step_counter++;

        auto it = tensor_cache.find(key);
        if (it != tensor_cache.end()) {
            it->second.last_use = step_counter;
            return it->second.tensor;
        }

        torch::Tensor tensor = recompute_func();
        size_t tensor_size = tensor.numel() * tensor.element_size();

        while (current_memory_usage + tensor_size > memory_limit && !tensor_cache.empty()) {
            evict_least_recently_used();
        }

        tensor_cache[key] = {tensor, recompute_func, tensor_size, step_counter};
        current_memory_usage += tensor_size;

        return tensor;
    }

private:
    void evict_least_recently_used() {
        auto lru_it = std::min_element(tensor_cache.begin(), tensor_cache.end(),
            [](const auto& a, const auto& b) { return a.second.last_use < b.second.last_use; });

        current_memory_usage -= lru_it->second.size;
        tensor_cache.erase(lru_it);
    }
};