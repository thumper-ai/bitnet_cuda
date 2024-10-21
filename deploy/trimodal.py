# ---
# deploy: true
# tags: ["use-case-lm-inference"]
# ---
# # Serverless TensorRT-LLM (LLaMA 3 8B)
#
# In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta's LLaMA 3 8B model
# at a total throughput of roughly 4,500 output tokens per second on a single NVIDIA A100 40GB GPU.
# At [Modal's on-demand rate](https://modal.com/pricing) of ~$4/hr, that's under $0.20 per million tokens --
# on auto-scaling infrastructure and served via a customizable API.
#
# Additional optimizations like speculative sampling and FP8 quantization can further improve throughput.
# For more on the throughput levels that are possible with TensorRT-LLM for different combinations
# of model, hardware, and workload, see the
# [official benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/71d8d4d3dc655671f32535d6d2b60cab87f36e87/docs/source/performance.md).
#
# ## Overview
#
# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the LLaMA 3 8B model.
#
# ### Build process
#
# Any given TensorRT-LLM service requires a multi-stage build process,
# starting from model weights and ending with a compiled engine.
# Because that process touches many sharp-edged high-performance components
# across the stack, it can easily go wrong in subtle and hard-to-debug ways
# that are idiosyncratic to specific systems.
# And debugging GPU workloads is expensive!
#
# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of a TensorRT-LLM
# build process that works on Modal.
#
# ### Engine configuration
#
# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive performance, but only if you tune it carefully.
# We carefully document the choices we made here and point to additional resources
# so you know where and how you might adjust the parameters for your use case.
#
# ## Installing TensorRT-LLM
#
# To run TensorRT-LLM, we must first install it. Easier said than done!
#
# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.
#
# We start from the official `nvidia/cuda:12.1.1-devel-ubuntu22.04` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

from typing import Optional

import modal
import pydantic  # for typing, used later

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)

# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.10.0.dev2024042300",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.
#
# End-to-end, this step takes five minutes.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run trtllm_llama.py`
# so that it runs in the background while you read the rest.
#
# ## Downloading the Model
#
# Next, we download the model we want to serve. In this case, we're using the instruction-tuned
# version of Meta's LLaMA 3 8B model.
# We use the function below to download the model from the Hugging Face Hub.

MODEL_DIR = "/root/model/model_input"
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"  # pin model revisions to prevent unexpected changes!


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()


# Just defining that function doesn't actually download the model, though.
# We can run it by adding it to the image's build process with `run_function`.
# The download process has its own dependencies, which we add here.

MINUTES = 60  # seconds
tensorrt_image = (  # update the image by downloading the model we're using
    tensorrt_image.pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "requests~=2.31.0",
    )
    .env(  # hf-transfer: faster downloads, but fewer comforts
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * MINUTES,
    )
)

# ## Configuring the model
#
# Now that we have the model downloaded, we need to convert it to a format that TensorRT-LLM can use.
# We use a convenience script provided by the TensorRT-LLM team.
# This script takes a few minutes to run.

GIT_HASH = "71d8d4d3dc655671f32535d6d2b60cab87f36e87"
CHECKPOINT_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/llama/convert_checkpoint.py"

# TensorRT-LLM requires that a GPU be present to load the model, even though it isn't used directly during this conversion process.
# We'll use a single A100-40GB GPU for this example, but we have also tested it successfully with A10G, A100-80GB, and H100 GPUs.
#
# The most important feature to track when selecting hardware to run on is GPU RAM:
# larger models, longer sequences, and bigger batches all require more memory,
# We tuned all three to maximize throughput on this example.
#
# The amount of GPU RAM on a single card is a tight constraint for most LLMs:
# RAM is measured in tens of gigabytes and
# models have billions of floating point parameters,
# each consuming one to four bytes of memory.
# The performance cliff if you need to spill to CPU memory is steep,
# so the only solution is to split the model across multiple GPUs.
# This is particularly important when serving larger models (e.g. 70B or 8x22B).

N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = modal.gpu.H100(count=N_GPUS)

# This is also the point where we specify the data type for this model.
# We use IEEE 754-compliant half-precision floats, (`float16`), because we found that it resulted in marginally higher throughput,
# but the model is provided in Google's
# [`bfloat16` format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).
# On the latest Ada Lovelace chips, you might use `float8` to reduce GPU RAM usage and speed up inference,
# but note that the FP8 format is very new, so expect rough edges.

DTYPE = "float16"

# We put that all together with another invocation of `.run_commands`.

CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = (  # update the image by converting the model to TensorRT format
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"wget {CHECKPOINT_SCRIPT_URL} -O /root/convert_checkpoint.py",
            f"python /root/convert_checkpoint.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
            + f" --tp_size={N_GPUS} --dtype={DTYPE}",
        ],
        gpu=GPU_CONFIG,  # GPU must be present to load tensorrt_llm
    )
)

# ## Compiling the engine
#
# TensorRT-LLM achieves its high throughput primarily by compiling the model:
# making concrete choices of CUDA kernels to execute for each operation.
# These kernels are much more specific than `matrix_multiply` or `softmax` --
# they have names like `maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148t_nt`.
# They are optimized for the specific types and shapes of tensors that the model uses
# and for the specific hardware that the model runs on.
#
# That means we need to know all of that information a priori --
# more like the original TensorFlow, which defined static graphs, than like PyTorch,
# which builds up a graph of kernels dynamically at runtime.
#
# This extra layer of constraint on our LLM service is precisely
# what allows TensorRT-LLM to achieve its high throughput.
#
# So we need to specify things like the maximum batch size and the lengths of inputs and outputs.
# The closer these are to the actual values we'll use in production, the better the throughput we'll get.

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_BATCH_SIZE = (
    128  # better throughput at larger batch sizes, limited by GPU RAM
)
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN} --max_output_len={MAX_OUTPUT_LEN}"

# There are many additional options you can pass to `trtllm-build` to tune the engine for your specific workload.
# You can find the document we used for LLaMA
# [here](https://github.com/NVIDIA/TensorRT-LLM/tree/66ef1df492f7bc9c8eeb01d7e14db01838e3f0bd/examples/llama),
# which you can use to adjust the arguments to fit your workloads,
# e.g. adjusting rotary embeddings and block sizes for longer contexts.
#
# We selected plugins that accelerate two core components of the model: dense matrix multiplication and attention.
# You can read more about the plugin options [here](https://fetch.ai/blog/advancing-llm-optimization).

PLUGIN_ARGS = f"--gemm_plugin={DTYPE} --gpt_attention_plugin={DTYPE}"

# We put all of this together with another invocation of `.run_commands`.

tensorrt_image = (  # update the image by building the TensorRT engine
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
            + f" --tp_size={N_GPUS} --workers={N_GPUS}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
)

# ## Serving inference at thousands of tokens per second
#
# Now that we have the engine compiled, we can serve it with Modal by creating an `App`.

app = modal.App(
    f"tritception", image=tensorrt_image
)

# Thanks to our custom container runtime system, even this
# large, many gigabyte container boots in seconds.
#
# At container start time, we boot up the engine, which completes in under 30 seconds.
# Container starts are triggered when Modal scales up your infrastructure,
# like the first time you run this code or the first time a request comes in after a period of inactivity.
#
# Container lifecycles in Modal are managed via our `Cls` interface, so we define one below
# to manage the engine and run inference.
# For details, see [this guide](https://modal.com/docs/guide/lifecycle-functions).


@app.cls(
    gpu=GPU_CONFIG,
    container_idle_timeout=10 * MINUTES,
)
class Model:
    @modal.enter()
    def load(self):
        """Loads the TRT-LLM engine and configures our tokenizer.

        The @enter decorator ensures that it runs only once per container, when it starts."""
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # LLaMA models do not have a padding token, so we use the EOS token
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.eos_token}
        )
        # and then we add it from the left, to minimize impact on the output
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )

    @modal.method()
    def generate(self, prompts: list[str], settings=None):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        import time

        if settings is None or not settings:
            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings[
            "max_new_tokens"
        ] = MAX_OUTPUT_LEN  # exceeding this will raise an error
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}🚀 Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            parsed_prompts, return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(
            f"{COLOR['HEADER']}Input tensors:{COLOR['ENDC']}", inputs_t[:, :8]
        )

        outputs_t = self.model.generate(inputs_t, **settings)

        outputs_text = self.tokenizer.batch_decode(
            outputs_t[:, 0]
        )  # only one output per input, so we index with 0

        responses = [
            extract_assistant_response(output_text)
            for output_text in outputs_text
        ]
        duration_s = (time.monotonic_ns() - start) / 1e9

        num_tokens = sum(
            map(lambda r: len(self.tokenizer.encode(r)), responses)
        )

        for prompt, response in zip(prompts, responses):
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)  # to avoid log truncation

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {num_prompts} on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return responses


# ## Calling our inference function
#
# Now, how do we actually run the model?
#
# There are two basic methods: from Python via our SDK or from anywhere, by setting up an API.
#
# ### Calling inference from Python
#
# To run our `Model`'s `.generate` method from Python, we just need to call it --
# with `.remote` appended to run it on Modal.
#
# We wrap that logic in a `local_entrypoint` so you can run it from the command line with
# ```bash
# modal run trtllm_llama.py
# ```
#
# For simplicity, we hard-code a batch of 128 questions to ask the model.


@app.local_entrypoint()
def main():
    questions = [
        # Profound questions for self-referential unit testing of the codebase implementing an LLM for accelerated execution with ternary weights and fast implementation
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) leverage (compute unified device architecture, parallel computation capabilities, optimization of calculations) to optimize (single-bit large language model inference)?",
        "Explain the process of quantizing neural network weights to single-bit in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) achieve lossless inference with 1.58-bit models on (graphics processing units, specialized parallel processors, hardware for handling complex computations)?",
        "Describe the custom (compute unified device architecture, parallel computation capabilities, optimization of calculations) kernels used in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) for efficient matrix operations.",
        "What role does kernel fusion play in improving (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s performance on (compute unified device architecture, parallel computation capabilities, optimization of calculations) architectures?",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) manage (graphics processing unit memory, high-speed memory for GPUs, volatile storage for computations) allocation and deallocation efficiently?",
        "Explain the implementation of rematerialization strategies in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) to optimize memory usage.",
        "How are integer mapping functions utilized in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) for vectorizable operations?",
        "Discuss methods (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) uses to prevent race conditions in multithreaded (compute unified device architecture, parallel computation capabilities, optimization of calculations) environments.",
        "How is data shuffling implemented in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) to enhance computational efficiency?",
        "Explain how kernelized operations are designed and utilized within (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "Describe the self-referential unit testing framework for (compute unified device architecture, parallel computation capabilities, optimization of calculations) kernels in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) handle error checking and debugging in complex (compute unified device architecture, parallel computation capabilities, optimization of calculations) kernels?",
        "What strategies does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) employ for load balancing across (compute unified device architecture) threads and blocks?",
        "Explain the role of warp-level programming in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) optimizations.",
        "How are bitwise operations implemented for single-bit model inference in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)?",
        "Discuss the challenges of implementing single-bit large language models on (compute unified device architecture) and how (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) addresses them.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) utilize shared memory to enhance performance of (compute unified device architecture) kernels?",
        "Explain the use of template metaprogramming in optimizing (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) code.",
        "How is thread divergence minimized in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) implementations?",
        "Describe how (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) tests (compute unified device architecture) kernels for correctness and performance.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) manage synchronization between threads and blocks in (compute unified device architecture)?",
        "Explain the importance of memory coalescing in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) code.",
        "How are reduction operations efficiently implemented in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) using (compute unified device architecture)?",
        "Discuss the use of (compute unified device architecture) streams in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) for concurrent kernel execution.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) implement kernel fusion to reduce memory bandwidth requirements?",
        "Explain how occupancy calculators are used to optimize (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) kernels.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) ensure numerical stability with single-bit quantization?",
        "Describe the approach for performance testing of (compute unified device architecture) kernels in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) handle graphics processing unit memory fragmentation over prolonged computations?",
        "Explain the use of dynamic parallelism in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) code.",
        "How are custom data types like packed ternary representations defined and used in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)?",
        "Discuss the implementation and testing of the custom memory manager in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) optimize graphics processing unit utilization for large-scale language models?",
        "Explain the profiling techniques used in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) for (compute unified device architecture) kernel optimization.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) integrate with deep learning frameworks for model deployment?",
        "Describe the process and challenges of compiling (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) with (NVIDIA CUDA Compiler, compiler for CUDA code, tool for compiling GPU code).",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) handle compatibility across different (NVIDIA graphics processing unit architectures, versions of GPU hardware, generations of NVIDIA GPUs)?",
        "Explain the use of inline parallel thread execution assembly for kernel optimization in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How are atomic operations employed in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) to maintain data integrity?",
        "Discuss latency reduction strategies in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s inference pipeline.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) implement single-bit gradient computation for potential training support?",
        "Explain methods for maximizing memory bandwidth utilization in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How are large matrix operations optimized in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) using (compute unified device architecture)?",
        "Describe the testing framework for verifying correctness of (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) code.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) utilize mixed-precision computations to improve performance?",
        "Discuss handling of overflows and underflows in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s quantized computations.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) handle model serialization and deserialization efficiently?",
        "Explain the impact of kernel launch parameters on (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s performance.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) maintain compatibility with different (compute unified device architecture toolkit versions, versions of CUDA software, updates of CUDA tools)?",
        "Describe methods for detecting memory leaks in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s graphics processing unit code.",
        "What techniques does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) use to optimize instruction-level parallelism?",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) handle batch processing and data loading efficiently?",
        "Explain the process for integrating and testing new (compute unified device architecture) kernels in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How are unit tests structured in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) to cover (compute unified device architecture) kernel functionality?",
        "Discuss challenges and solutions for debugging (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) code.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) implement logging and error reporting for (compute unified device architecture) operations?",
        "Explain potential extensions of (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) for future graphics processing unit architectures.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) manage and test multi-graphics processing unit configurations effectively?",
        "Describe the continuous integration pipeline for (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) codebase.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) mitigate precision loss due to aggressive quantization?",
        "Discuss future optimization paths for (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) using advanced (compute unified device architecture) features.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) ensure reproducible results across different hardware setups?",
        "Explain benchmarking methods used to compare (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) with other inference frameworks.",
        "How are code documentation and comments maintained in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s (compute unified device architecture) code?",
        "What methods are used in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) to detect race conditions and deadlocks?",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) improve code coverage through unit tests for (compute unified device architecture) code?",
        "Discuss the use of (compute unified device architecture) Graphs in potential optimizations for (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation).",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s implementation guide improvements in (Triton-based large language models, high-performance LLMs using Triton, optimized LLMs with Triton backend)?",
        "Explain the role of self-referential unit tests in enhancing (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s reliability.",
        "How does (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) integrate self-testing mechanisms within its codebase?",
        "Discuss the importance of self-referential tests for optimizing (Triton-based large language models, high-performance LLMs using Triton, optimized LLMs with Triton backend).",
        "How can insights from (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation)'s code inform the development of more efficient (Triton-based large language models)?",
        "Explain how unit tests in (large language model implementation for accelerated execution, ternary quantization of weights, high-performance implementation) contribute to the overall system robustness.",
    ]

    model = Model()
    model.generate.remote(questions)


# ### Calling inference via an API
#
# We can use `modal.web_endpoint` and `app.function` to turn any Python function into a web API.
#
# This API wrapper doesn't need all the dependencies of the core inference service,
# so we switch images here to a basic Linux image, `debian_slim`, which has everything we need.

web_image = modal.Image.debian_slim(python_version="3.10")

# From there, we can take the same remote generation logic we used in `main`
# and serve it with only a few more lines of code.


class GenerateRequest(pydantic.BaseModel):
    prompts: list[str]
    settings: Optional[dict] = None


@app.function(image=web_image)
@modal.web_endpoint(
    method="POST", label=f"tritception", docs=True
)
def generate_web(data: GenerateRequest) -> list[str]:
    """Generate responses to a batch of prompts, optionally with custom inference settings."""
    return Model.generate.remote(data.prompts, settings=None)


# To set our function up as a web endpoint, we need to run this file --
# with `modal serve` to create a hot-reloading development server or `modal deploy` to deploy it to production.
#
# ```bash
# modal serve trtllm_llama.py
# ```
#
# The URL for the endpoint appears in the output of the `modal serve` or `modal deploy` command.
# Add `/docs` to the end of this URL to see the interactive Swagger documentation for the endpoint.
#
# You can also test the endpoint by sending a POST request with `curl` from another terminal:
#
# ```bash
# curl -X POST url-from-output-of-modal-serve-here \
# -H "Content-Type: application/json" \
# -d '{
#     "prompts": ["Tell me a joke", "Describe a dream you had recently", "Share your favorite childhood memory"]
# }' | python -m json.tool # python for pretty-printing, optional
# ```
#
# And now you have a high-throughput, low-latency, autoscaling API for serving LLaMA 3 8B completions!
#
# ## Footer
#
# The rest of the code in this example is utility code.


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text
