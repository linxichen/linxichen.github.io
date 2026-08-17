---
title: "Fitting Qwen3.8-27B with 128K Context on a 24GB TITAN RTX: A Self-Hosting Tuning Journey"
date: 2026-08-17T12:30:00-04:00
categories:
  - Tutorial
tags:
  - self-hosting
  - llama-cpp
  - qwen
  - homelab
  - llm
---

My agent (Hermes, running on a Mac mini in the house) was "hanging" whenever it talked to my freshly deployed local model server. The server was up, health checks passed, small requests worked — but the moment a real conversation landed, nothing came back for eight minutes. This post is the story of diagnosing that, and the arithmetic that turned a 116 tok/s server into a 600 tok/s (7,752 tok/s cached) server on the same hardware.

## The setup

- **Host**: a Linux box on my LAN (16 cores, 64 GB RAM)
- **GPU**: NVIDIA TITAN RTX — 24 GB VRAM, Turing (compute capability 7.5)
- **Model**: `unsloth/Qwen3.8-27B-GGUF` at **UD-Q4_K_XL** (~17.9 GB on disk)
- **Server**: `llama-server` from llama.cpp, serving an OpenAI-compatible API at `http://<lan-ip>:8080/v1`

Qwen3.8-27B is a native vision-language model with a 262K context window. I wanted it local, wired into my agent as a named provider ("LinxiCloud"), so my chat traffic stops depending on any cloud API.

## The symptom

The agent sent a routine ~54,000-token prompt (system prompt + tools + conversation history — agents are prompt-heavy by nature). The server log told the story:

```
slot print_timing: prompt processing, n_tokens = 51585, progress = 0.45, t = 480.14 s / 107.44 tokens per second
slot release: task 25 | stop processing: n_tokens = 53633
```

**~107 tokens per second of prompt processing.** At that speed, a 54K prompt takes over eight minutes *before the first generated token*. My agent's HTTP client had long since timed out, retried, and given up. The server was never broken — it was just too slow to look alive.

## First wrong theory: Vulkan vs CUDA

llama.cpp stopped shipping prebuilt Linux CUDA binaries, so my first launch used the official **Vulkan** build. A friend asked the obvious question: "wait, is it not running on CUDA? I see it in nvidia-smi."

Good question, and worth being precise about: **Vulkan compute still runs on the GPU.** `nvidia-smi` showed the process and ~21.6 GB of VRAM in use the whole time. CUDA vs Vulkan is about *which kernel API llama.cpp uses*, not *whether* the GPU is used. Still, CUDA kernels are usually noticeably faster, so I did the upgrade properly:

1. `sudo apt install nvidia-cuda-toolkit cmake` (toolkit 12.0; driver reports 12.8 — same major version, fine)
2. `git clone --depth 1 https://github.com/ggml-org/llama.cpp`
3. `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` ← Turing
4. `cmake --build build -j16` (~10 minutes on 16 cores)
5. `sudo cmake --install build --prefix /usr/local`, plus one gotcha: the installed `llama-server` couldn't find `libllama-server-impl.so` until I added `/usr/local/lib` to the loader config and ran `sudo ldconfig`.

Then I re-ran the same 27K-token benchmark. Result: **116 tok/s. Identical to Vulkan.**

CUDA was never the bottleneck. Time to do the math.

## The actual bottleneck: KV cache doesn't fit in VRAM

### What the KV cache is

A transformer's attention layer, when generating token *N*, needs to look at the keys and values of **every previous token**. Recomputing those from scratch each step would be quadratic madness, so inference engines cache them: the **KV cache** stores, for every token in the context and every attention layer, a K vector and a V vector.

> **Per-request control**: the `--chat-template-kwargs` flag only sets the server *default*. Clients can override reasoning effort per request with the OpenAI-style `reasoning_effort` field (`"none"`, `"low"`, `"medium"`, `"high"`) — verified against the live server: `none` produces zero thinking tokens, `low` produces a short trace. Agentic clients that want cheap fast turns can send `"none"` while keeping deep thinking for hard prompts.

The cost per token is:

```
KV bytes/token = 2 (K and V) × n_kv_heads × head_dim × n_attention_layers × bytes_per_element
```

For Qwen3.8-27B specifically, the architecture is hybrid — and this is the fun wrinkle. Of its 64 layers, **48 are Gated DeltaNet** (linear attention with a *constant-size* recurrent state — it doesn't grow with context at all) and only **16 are full Gated Attention** layers with 4 KV heads of dimension 256. So:

```
16 layers × 2 × 4 heads × 256 dim = 32,768 elements per token
                                        = 64 KiB/token at f16
                                        ≈ 34 KiB/token at q8_0
```

That hybrid design is why this model's long context is even *approachable* on consumer VRAM — a dense 64-layer model with the same head config would cost 4× more.

### The budget

The TITAN RTX has 24,576 MiB, of which about **23.8 GB is usable**. The bill:

| Item | Size |
|---|---|
| Model weights (UD-Q4_K_XL) | ~17.9 GB |
| Compute buffers (CUDA/graph overhead) | ~1.2–1.5 GB |
| **Left for KV cache** | **~4.5 GB** |

Now the KV cache demands:

| Context | f16 KV | q8_0 KV |
|---|---|---|
| 262,144 (full window) | ~17.2 GB | ~9.1 GB |
| 131,072 | ~8.6 GB | **~4.6 GB** ✓ |

There it is. My original launch used `-c 262144` with default f16 KV — a **17.2 GB** cache demand on top of a 17.9 GB model. Total ~36 GB against 24 GB of VRAM. llama.cpp doesn't refuse; it silently overflows the excess into system RAM (the process RSS confirmed it: ~20 GB of host memory). Every attention pass during prompt processing then straddles PCIe, and throughput collapses to ~116 tok/s — on CUDA or Vulkan alike. The backend was innocent; the *spill* was guilty.

### The fix

Drop the context to 128K (still enormous for real conversations) and quantize the KV cache to `q8_0` (8-bit — quality impact on KV is well under measurement noise for chat/agentic use):

```bash
/usr/local/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \
  --mmproj ~/llama.cpp/mmproj-F16.gguf \
  --host <lan-ip> \
  -c 131072 \
  -ngl 99 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

Flag by flag:

- **`-c 131072`** — context sized so the KV cache *plus* model *plus* compute fits VRAM with margin, per the table above. Observed total: 23.1 GB of 24 GB.
- **`-ngl 99`** — all 64 layers offloaded to the GPU; nothing splits to CPU.
- **`-fa on`** — FlashAttention. Required if you quantize the V cache (older `-fa` boolean syntax is gone in current builds — it now takes `on|off|auto`).
- **`--cache-type-k/--cache-type-v q8_0`** — halves KV bytes per token (64 → ~34 KiB).
- **`--mmproj`** — the vision projector (this is a VLM; note the path must point at a real local file — `-hf repo/mmproj-F16.gguf` inside `--mmproj` failed with a confusing "file does not exist" on my build, so I downloaded it explicitly).
- **`--chat-template-kwargs '{"reasoning_effort":"medium"}'`** — Qwen3.8 thinks by default at "xhigh"; medium is the right speed/quality tradeoff for an agent workhorse.

## Results

Same 27,220-token benchmark:

| Config | Prompt processing |
|---|---|
| 262K ctx, f16 KV, Vulkan | ~116 tok/s |
| 262K ctx, f16 KV, CUDA | ~116 tok/s (spill-bound) |
| **131K ctx, q8_0 KV, CUDA, FA** | **600 tok/s cold, 7,752 tok/s cached** |

That last number deserves explanation: llama.cpp keeps a **prompt cache** of processed tokens per slot. When the same prefix arrives again (an agent resends its system prompt + history with one new message appended), only the delta is processed. The 54K prompt that started this whole saga went from *eight minutes* to **45 seconds cold and ~3.5 seconds warm**. The agent, in other words, now gets answers faster than I type follow-ups.

## Bonus round: MTP speculative decoding, 20 → 30 tok/s

Prompt processing was now fine, but token generation sat at ~20 tok/s — a dense 27B on a TITAN RTX is decode-bandwidth-bound (672 GB/s divided by ~18 GB of weights you touch per token). You can't buy more bandwidth, but Qwen3.8 was trained with an **MTP (multi-token prediction) head**, and llama.cpp supports it natively via `--spec-type draft-mtp`. The idea: a small draft model predicts the next few tokens cheaply, the big model verifies them in one batched pass, and accepted drafts are free tokens. Acceptance ~55% with mean accepted length 2.65 gave a **1.5× decode speedup: ~30 tok/s** measured end-to-end.

The catch on 24 GB: the draft head needs VRAM too. The Q8 MTP head (3.2 GB) plus full buffers didn't fit at 131K context, and llama.cpp's default of `--parallel 4` was silently multiplying the KV cache 4×. Final working recipe: **96K context, `--parallel 1`, Q4_0 MTP draft head (1.7 GB)** — 23.6 GB total, ~30 tok/s single-stream. Tradeoffs: 35K less context and one slot (concurrent requests queue instead of interleaving), which is the right trade for a single-agent workload.

```bash
llama-server -m <model.gguf> \
  --mmproj <mmproj-F16.gguf> \
  --host <lan-ip> \
  -c 98304 --parallel 1 \
  -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-model mtp-Qwen3.8-27B-Q4_0.gguf
```

## Lessons worth keeping

1. **GPU usage ≠ GPU-resident.** `nvidia-smi` showing the process and VRAM in use tells you nothing about whether attention is spilling to system RAM. Check the process RSS against the model size, and llama.cpp's buffer logs for "CUDA0" vs "CPU" assignments.
2. **Do the KV arithmetic before choosing `-c`.** `bytes/token = 2 × kv_heads × head_dim × attn_layers × dtype_bytes`, and check hybrid architectures — DeltaNet/linear-attention layers often don't grow with context at all, which changes the calculus dramatically.
3. **The backend wasn't the problem.** CUDA is still worth having (and the source build is 15 minutes with `cmake --install` + `ldconfig`), but a VRAM overflow pins you at CPU speed no matter which kernel API renders the matmuls.
4. **Quantize the KV cache fearlessly.** q8_0 KV is essentially free accuracy-wise and literally doubles the context you can afford.
5. **Server "not responding" usually means "responding slowly."** Health checks and tiny curls pass; the failure only shows at production prompt sizes. Benchmark with a *real* 27K+ prompt before declaring victory.
6. **Prebuilt Linux CUDA binaries are gone** from llama.cpp releases — budget 15 minutes for the source build, `sudo cmake --install --prefix /usr/local`, and remember `ldconfig`.

The server now backs a named provider in my agent's config — any session can switch to it with `/model custom:LinxiCloud:...` — and the whole stack (quantized weights, KV budget, flash attention, prompt caching) fits in one aging-but-game Turing card. Self-hosting a frontier-adjacent 27B VLM at agent-scale context sizes on 2018 silicon is, it turns out, mostly an accounting exercise.

## What would 2× DGX Spark buy?

Natural question after all this VRAM accounting: what happens on modern hardware? NVIDIA's DGX Spark (formerly Digit) packs a GB10 Grace Blackwell Superchip with **128 GB unified LPDDR5x memory** and claims ~1 PFLOP FP4 / ~100 TFLOPS FP8 dense. Two of them networked via ConnectX-7 (200 Gbps) give you 256 GB of coherent memory. Re-running the same arithmetic:

**Weights**: The model is the same ~17.9 GB — but now there's zero pressure to stay at 4-bit. You could serve the **Q8 / UD-Q8_0 quant (~30 GB)** or even BF16 (~54 GB) and gain real quality, or serve *multiple* models side by side.

**Full KV cache, unquantized**: The headline number. KV at f16 is ~64 KiB/token (32,768 elements × 2 bytes). With 128 GB per node, budget ~85–95 GB for KV after weights and buffers on a single Spark:

| Config | KV budget | Context @ f16 (full precision) |
|---|---|---|
| 1× Spark, UD-Q4_K_XL (17.9 GB) | ~95 GB | ~1.55M tokens |
| 1× Spark, BF16 (54 GB) | ~60 GB | ~985K tokens |
| 2× Spark, UD-Q4_K_XL (tensor-parallel) | ~200 GB | ~3.3M tokens |

So two Sparks tensor-parallel the same Q4 model with **over 3 million tokens of full-precision KV cache** — the entire 262K native window (17.2 GB at f16) fits with 10× headroom, and you could serve contexts the model card doesn't even advertise. Even a single Spark runs the full 262K window at f16 KV with room to spare.

**Time to first token**: This is where it gets nuanced, because TTFT has two parts. The *compute* side (prompt processing throughput) improves with Blackwell's tensor cores and higher aggregate FLOPS — expect **2–4× the TITAN RTX's 600 tok/s**, so a 27K prompt drops from 45s cold to roughly 10–20s. But LPDDR5x memory bandwidth is the ceiling for token generation: ~273 GB/s per Spark versus the TITAN RTX's 672 GB/s GDDR6. On a memory-bandwidth-bound 27B decode (a single Streamline generation reads all active weights per token), a Spark may generate *slower per token* than the old Turing card. The honest summary: **TTFT on big prompts — much better. Sustained single-stream decode — roughly similar or slightly worse.** The wins are capacity (huge contexts, better quants, multiple models) and batch throughput, not single-stream tok/s.

For an agent workload like mine — giant prompts, short answers — that trade is exactly right: the 3.5s warm-cache turn latency would persist (it's compute-bound on the delta), the 8-minute disaster becomes a 10–20s worst case even cold, and reasoning models that think for hundreds of tokens per turn would benefit from batching multiple concurrent sessions across both nodes. And if 3M-token contexts sound absurd: they're the difference between an agent that forgets the start of a codebase review by the end, and one that doesn't.

*Estimates are back-of-envelope from published specs (128 GB LPDDR5x, ~273 GB/s, GB10 FLOPS claims); real numbers depend on llama.cpp GB10 support maturity, tensor-parallel efficiency over ConnectX-7, and MoE expert-routing patterns.*
