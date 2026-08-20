# Scaling Lab

A decoding LLM's real bottleneck is a memory read: every token requires streaming the
full weights through the GPU. This repo studies the two orthogonal levers that beat that
wall — **amortize the read** (speculative decoding: more tokens per pass) and **shard
the read** (tensor parallelism: less model per GPU) — both built from first principles,
both benchmarked, and both with honest negative results where the technique *loses*.

| Lever | Idea | Notebook |
|---|---|---|
| **Speculate** | cheap proposer drafts K tokens; target verifies all of them in one weight read; rejection sampling keeps the output distribution *exactly* intact | [`speculative_decoding.ipynb`](speculative_decoding.ipynb) |
| **Shard** | split every weight matrix across GPUs (Megatron-style column/row parallelism); pay one all-reduce per matmul sandwich | [`tensor_parallelism.ipynb`](tensor_parallelism.ipynb) |

## Notebook 1 — Free Tokens: Speculative Decoding

The full arc: the roofline argument (7B bf16 ≈ 1 FLOP/byte vs the H100's ~300 FLOP/byte
ridge → decode is bandwidth-bound), the draft-and-verify algorithm, the rejection-sampling
proof that output quality is mathematically untouched (the min/residual terms telescope
back to exactly p(x)), a runnable **from-scratch implementation** (GPT-2 drafting for
GPT-2-large, with measured acceptance rate), the **full proposal zoo** — draft models,
n-gram/prompt-lookup, suffix decoding (local + global suffix trees), Medusa's
tree-verified prediction heads, and the three EAGLE generations including EAGLE-3's
training-time test — each explained and scored on the ratio that decides everything
(*tokens accepted per verify ÷ proposal cost*), a **vLLM benchmark harness** that
launches a fresh server per configuration, and a production guide for choosing and
tuning a method (K, temperature, memory headroom, when to skip speculation entirely).

**Measured (Qwen2.5-7B-Instruct, NVIDIA H100 80GB, vLLM 0.19.0, greedy, 256 tok × 10 prompts):**

| Config | Throughput | Latency | Speedup |
|---|---:|---:|---:|
| baseline | 162.1 tok/s | 6.17 ms/tok | 1.00× |
| n-gram (K=5) | 139.8 tok/s | 7.16 ms/tok | **0.86× — slower** |
| EAGLE-3 (K=3) | 334.6 tok/s | 2.99 ms/tok | **2.06×** |

Three findings worth internalizing: the baseline lands at 68% of the 239 tok/s bandwidth
roofline (which is what a *healthy* baseline looks like); n-gram speculation **loses 14%**
on open-ended prompts because free-but-weak proposals still pay verification bookkeeping;
and EAGLE-3's per-prompt spread (264–425 tok/s) tracks text predictability exactly as
acceptance-rate theory predicts. Speculation is a workload decision, not a checkbox.

## Notebook 2 — Splitting the Matmul: Tensor Parallelism

Megatron-style TP from nothing: a 29M-parameter GPT, column-parallel and row-parallel
linear layers with their conjugate autograd communication rules (identity-forward /
all-reduce-backward and vice versa), the head-split attention + d_ff-split MLP sharding
map, `torchrun` benchmarks, a shard inspector, and two microbenchmarks — all-reduce
latency vs message size, and a **compute-vs-communication crossover sweep** across
`d_model`.

**Measured (2× NVIDIA H200, NCCL/NVLink):**

| Metric | 1 GPU | TP=2 |
|---|---:|---:|
| Params per GPU | 29.4M | 17.4M (−40.8%) |
| Peak memory | 1,204 MB | 837 MB (−30.5%) |
| Step time | 11.84 ms | 19.19 ms |
| **Speedup** | 1× | **0.62× — slower** (31% efficiency) |

This is the tutorial's centerpiece, not its embarrassment: at `d_model=512` the sharded
matmul costs 24 µs while its all-reduce costs 46 µs, and the all-reduce latency floor
(~105 µs even for kilobyte messages) means twelve of them per forward simply cannot be
amortized by a 29M-parameter model. Sharding also isn't 1/N: LayerNorms, embeddings, and
every residual stay replicated at full width. **TP's first job is memory — it makes
models fit; speed is what it charges for that**, and the crossover sweep shows
efficiency climbing as matmuls grow into the interconnect latency (a Llama-70B layer
does ~4,000× more work per all-reduce than this toy).

## Running it

```bash
pip install -r requirements.txt

# Notebook 1: from-scratch demo runs on any CUDA GPU (or CPU, slowly).
# The vLLM section needs an 80GB-class GPU and `pip install vllm`;
# measured results are embedded so the analysis reads standalone.
jupyter lab speculative_decoding.ipynb

# Notebook 2: needs 2+ CUDA GPUs (scripts are written and launched via torchrun).
jupyter lab tensor_parallelism.ipynb
```

## References

- Shoeybi et al., *Megatron-LM* — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Narayanan et al., *Efficient Large-Scale LM Training on GPU Clusters* — [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Leviathan et al., *Fast Inference via Speculative Decoding* — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., *Accelerating LLM Decoding with Speculative Sampling* — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Kwon et al., *PagedAttention / vLLM* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Cai et al., *Medusa* — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Li et al., *EAGLE* — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) · *EAGLE-2* — [arXiv:2406.16858](https://arxiv.org/abs/2406.16858) · *EAGLE-3* — [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)

Companion repos: [kv-cache-lab](https://github.com/RishikeshAluguvelli/kv-cache-lab)
(cache latency, memory, quantization) ·
[attention-lab](https://github.com/RishikeshAluguvelli/attention-lab)
(MHA/GQA/MQA/MLA and FlashAttention benchmarks).
