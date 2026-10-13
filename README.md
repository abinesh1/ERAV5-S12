# ZeRO 1 / 2 / 3 on 32 Virtual GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abinesh1/ERAV5-S12/blob/main/ZeRO_32_GPU_Simulation.ipynb)

A from-scratch simulation of **DeepSpeed ZeRO** stages 1, 2 and 3 across **32 virtual GPUs**, in
NumPy, with no distributed runtime and no GPU required. One small MLP is trained four times — plain
data parallelism plus each ZeRO stage — and the notebook **measures** what actually changes: bytes
resident per GPU, peak memory, bytes on the wire, and FLOPs.

The headline result is the one ZeRO promises: **the same training run, bit for bit, in 32x less memory
per GPU.**

📓 **[`ZeRO_32_GPU_Simulation.ipynb`](ZeRO_32_GPU_Simulation.ipynb)** — the whole thing, self-contained,
with outputs committed. Runs top to bottom in about a minute.

---

## Results at a glance

All figures measured at `N = 32`, `Ψ = 399,370` parameters, mixed-precision Adam.

| | Baseline DP | ZeRO-1 (P<sub>os</sub>) | ZeRO-2 (P<sub>os+g</sub>) | ZeRO-3 (P<sub>os+g+p</sub>) |
|---|---|---|---|---|
| **Memory / GPU** | 6.094 MB | 1.666 MB | 0.928 MB | **0.190 MB** |
| **vs baseline** | 1.00x | 3.66x less | 6.56x less | **32.00x less** |
| Peak (incl. transients) | 6.658 MB | 2.230 MB | 1.493 MB | 1.256 MB |
| **Communication / step** | 1.94x | 1.94x | 1.94x | **2.91x** |
| **Matmul FLOPs / GPU / step** | 76.5 M | 76.5 M | 76.5 M | 76.5 M |
| **Adam updates / GPU / step** | 399,392 | 12,481 | 12,481 | **12,481 (32x less)** |
| Final loss | 0.371121 | 0.371121 | 0.371121 | 0.371121 |

Communication is expressed as a multiple of the fp16 model size. The values land at `1.94x`/`2.91x`
rather than `2x`/`3x` because a ring collective moves `(N-1)/N` of the payload, and `31/32 = 0.969`.

![equivalence](figures/fig1_equivalence.png)

Read the left panel carefully: there are four curves there, drawn at decreasing line widths so the
overlap is visible. They coincide **exactly** — `max |loss − baseline| = 0.000e+00` at every step, for
all three ZeRO stages.

---

## Why `16Ψ`, and what ZeRO deletes

Mixed-precision Adam keeps six tensors alive per parameter:

| tensor | dtype | bytes/param |
|---|---|---|
| parameters (compute copy) | fp16 | 2 |
| gradients | fp16 | 2 |
| parameters (fp32 master copy) | fp32 | 4 |
| Adam momentum `m` | fp32 | 4 |
| Adam variance `v` | fp32 | 4 |
| | | **16Ψ** |

Under plain data parallelism **every GPU holds all 16Ψ of it**. Thirty-two GPUs hold thirty-two
identical copies. That redundancy buys nothing, because the Adam update is *elementwise*: parameter
`j`'s new value depends only on `g[j]`, `m[j]`, `v[j]`, `master[j]`. Rank 7 has no reason to own the
momentum for a parameter rank 7 is not updating.

ZeRO deletes the redundancy one tier at a time:

| stage | shards | memory / GPU | comm |
|---|---|---|---|
| Baseline DP | nothing | `16Ψ` | `2Ψ` |
| **ZeRO-1** | optimizer states (12 bytes) | `4Ψ + 12Ψ/N` | `2Ψ` |
| **ZeRO-2** | \+ gradients | `2Ψ + 14Ψ/N` | `2Ψ` |
| **ZeRO-3** | \+ parameters | `16Ψ/N` | `3Ψ` |

![memory](figures/fig2_memory.png)

The stacked bars make the tiers obvious. The green optimizer block is 75% of the baseline and ZeRO-1
flattens it alone. ZeRO-2 then takes the gradients, leaving `2Ψ` of fp16 parameters as a floor. ZeRO-3
removes the floor.

Measured per-component, in MB per GPU:

| | params | grads | optimizer | total |
|---|---|---|---|---|
| Baseline DP | 0.762 | 0.762 | 4.571 | 6.094 |
| ZeRO-1 | 0.762 | 0.762 | 0.143 | 1.666 |
| ZeRO-2 | 0.762 | 0.024 | 0.143 | 0.928 |
| ZeRO-3 | 0.024 | 0.024 | 0.143 | 0.190 |

---

## How the simulation works

### The 32 virtual GPUs

Each virtual GPU is a `VGPU` object owning its own NumPy arrays. The 32 of them execute
**sequentially** in one process, bulk-synchronous: every rank finishes layer *i* before any rank starts
layer *i+1*, which is precisely where a real collective would sit.

**What is real:** the sharding (each rank physically allocates only its slice — `state_bytes()` sums
`.nbytes` of arrays that actually exist), the collectives (data genuinely moves between 32 separate
arrays), the byte counts, and the arithmetic.

**What is not real:** wall-clock parallelism. Python threads would not fix this — the GIL serialises
them anyway — and a speedup number derived from 32 GIL-bound threads would be a worse lie than not
reporting one. So the notebook reports **FLOPs and communicated bytes**, which are what a distributed
run is actually bounded by, and never claims a speedup.

### One flat parameter vector

Like real DeepSpeed, all parameters are flattened into a single contiguous buffer, padded to a
multiple of `N`, and *that* is sharded — rank `r` owns `[r·S, (r+1)·S)`. This keeps every shard exactly
equal in size. Shard boundaries consequently cut straight through layers, which is fine and is what
DeepSpeed does.

```
Ψ = 399,370 params  ->  padded 399,392  ->  12,481 elements per GPU

  layer 0   256 x 512   flat[      0 : 131,584]   32.9% of the model
  layer 1   512 x 512   flat[131,584 : 394,240]   65.8% of the model
  layer 2   512 x  10   flat[394,240 : 399,370]    1.3% of the model
```

Each layer still occupies one contiguous **bucket**, and the bucket is the unit that gets all-gathered
and reduce-scattered.

### Three collectives, with a byte meter

| collective | result | bytes moved per rank |
|---|---|---|
| `all_reduce(x)` | every rank gets the mean | `2·(N−1)/N·size(x)` |
| `reduce_scatter(x)` | rank `r` gets the mean, restricted to its shard | `(N−1)/N·size(x)` |
| `all_gather(shards)` | every rank gets the concatenation | `(N−1)/N·size(out)` |

Every reduction accumulates in fp32 in a **fixed rank order** and rounds to fp16 at the end, in all
four configurations. That determinism is what makes the bitwise-equivalence check possible.

### Where the stages actually differ

The entire stage-specific behaviour lives in two functions:

- **`layer_params(i)`** — how a rank obtains layer `i`'s weights. Stages 0–2 already hold them
  (no communication). Stage 3 must **all-gather** the bucket from all 32 shards, use it, and drop it —
  in forward *and* again in backward. That is where ZeRO-3's extra `Ψ` of traffic comes from.
- **`deposit_grad(i)`** — what a rank does with the gradient it just produced. Stages 0–1 store the
  whole thing. Stages 2–3 **reduce-scatter** the bucket immediately and keep only their own slice,
  which is what makes gradient memory `Ψ/N` instead of `Ψ`.

Collectives actually issued per step, counted by the meter:

| | collectives per step | volume |
|---|---|---|
| Baseline DP | 1 all-reduce | `2Ψ` |
| ZeRO-1 | 1 reduce-scatter + 1 all-gather | `Ψ + Ψ` |
| ZeRO-2 | 3 reduce-scatter (one per layer) + 1 all-gather | `Ψ + Ψ` |
| ZeRO-3 | 6 all-gather (3 fwd + 3 bwd) + 3 reduce-scatter | `Ψ + Ψ + Ψ` |

---

## Communication: what ZeRO-3 costs

![comm](figures/fig3_comm.png)

**ZeRO-1 and ZeRO-2 are free.** An all-reduce *is* a reduce-scatter followed by an all-gather; ZeRO
simply performs the optimizer step between the two halves instead of after both. Identical bytes, far
less memory. There is no reason not to enable them.

**ZeRO-3 is the only stage that pays: +50% traffic for 32x less memory.** In a real cluster much of
that gather is hidden behind compute by prefetching layer `i+1` while layer `i` runs.

## Computation: the model's arithmetic is unchanged, the optimizer step shrinks by N

Matmul FLOPs per GPU are **identical across all four configurations** (76.5 MFLOP/step here — the
notebook asserts it). Every rank still performs a full forward and backward over its own micro-batch,
and nothing is recomputed. ZeRO redistributes *state*, not the model's arithmetic.

The one piece of arithmetic that does change is the **optimizer step**. Under plain data parallelism
every rank runs Adam over all `Ψ` parameters, so 32 GPUs compute the same update 32 times. From ZeRO-1
on, each rank updates only the `Ψ/N` shard whose optimizer state it owns:

| | Adam updates / GPU / step | across all 32 GPUs |
|---|---|---|
| Baseline DP | 399,392 (`Ψ`) | 12,780,544 — 32 redundant copies |
| ZeRO-1 / 2 / 3 | 12,481 (`Ψ/N`) | 399,392 — each parameter updated once |

Adam is elementwise (`O(Ψ)`) and small next to the `6·B·Ψ` of forward+backward, so this is a bonus
rather than the headline, but it is measured and asserted, not assumed.

This is exactly what separates ZeRO from tensor/model parallelism, which *does* split the arithmetic
and therefore needs high-bandwidth intra-node interconnect to be viable. ZeRO's communication is
bandwidth-bound bulk transfer that tolerates ordinary inter-node networks.

## Peak is not persistent

ZeRO-3's persistent state is `16Ψ/N`, but its **peak** is higher, because to multiply by layer `i` a
rank must briefly hold *all* of layer `i`, not `1/N` of it:

```
peak ≈ 16Ψ/N  +  (largest layer)  +  (its gradient bucket)  +  activations
```

At this toy scale the transient **dominates** — measured at 6.6x the persistent state, because the
largest layer is 65.8% of a 3-layer model, which is enormous next to `1/32` of it. That is not a
simulation artifact; it is why production ZeRO-3 prefetches, buckets below layer granularity, and is
paired with activation checkpointing. It is also why ZeRO-3's advantage *grows with depth*: in an
80-layer transformer one gathered layer is 1.25% of the model, not 66%.

Activations are measured too (0.063 MB/GPU) and are **identical in all four configurations** — ZeRO
shards model state, not activations. Addressing those is a different technique.

---

## Scaling with world size

![scaling](figures/fig4_scaling.png)

Memory per GPU in MB, measured by constructing ranks at each world size:

| N | Baseline | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 1 | 6.094 | 6.094 | 6.094 | 6.094 |
| 2 | 6.094 | 3.809 | 3.428 | 3.047 |
| 4 | 6.094 | 2.666 | 2.095 | 1.523 |
| 8 | 6.094 | 2.095 | 1.428 | 0.762 |
| 16 | 6.094 | 1.809 | 1.095 | 0.381 |
| 32 | 6.094 | 1.666 | 0.928 | **0.190** |

The shape of each curve is the whole story:

- **Baseline is flat.** Adding GPUs adds throughput and exactly zero memory headroom.
- **ZeRO-1 asymptotes to `4Ψ`** — the replicated params and grads it never touches.
- **ZeRO-2 asymptotes to `2Ψ`.**
- **ZeRO-3 has no floor.** It is the only stage that sits on the ideal `1/N` line, and the only one
  where buying more GPUs means you can train a *bigger* model rather than just a faster one.

## What this means at real scale

The closed-form accounting is **asserted against the measured allocations** in the notebook, at every
world size above — which is what licenses extrapolating it to sizes that will not fit in a notebook.

GB per GPU at `N = 32`, model states only (activations extra), against an 80 GB H100:

| model | params | Baseline | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| GPT-2 XL | 1.5B | 24 | 6.56 | 3.66 | 0.75 |
| ZeRO paper reference | 7.5B | 120 ❌ | 32.8 | 18.3 | 3.75 |
| Llama-3 70B | 70B | 1120 ❌ | 306 ❌ | 171 ❌ | 35.0 |
| GPT-3 175B | 175B | 2800 ❌ | 766 ❌ | 427 ❌ | 87.5 ❌ |

![projection](figures/fig5_projection.png)

- A **7.5B** model needs 120 GB/GPU under plain data parallelism — it fits on nothing. ZeRO-1 alone,
  which costs *no extra communication at all*, brings it to ~33 GB and it trains on 32 GPUs.
- **70B** is out of reach for every stage except ZeRO-3, which lands at 35 GB.
- **175B** does not fit at `N = 32` even with ZeRO-3 (87.5 GB). The answer is more GPUs — `16Ψ/N` keeps
  falling — or ZeRO-Infinity's offload to CPU/NVMe. Nothing here is a free lunch: ZeRO-3 converts a
  memory problem into a bandwidth problem, and eventually you must buy more of one.

---

## Correctness: why you should believe the memory numbers

A memory simulation that does not train correctly is just a spreadsheet. Four assertions in the
notebook keep it honest, and the notebook fails to execute if any of them breaks:

1. **Bitwise equivalence.** All four configurations, from the same seed on the same data, produce
   identical loss curves: `max |loss − baseline| == 0.0`. If ZeRO-3 skipped a gather or sharded the
   optimizer incorrectly, this would drift immediately.
2. **Measured == theory.** `state_bytes()` sums real `.nbytes`, and every value is asserted equal to
   the paper's `16Ψ` / `4Ψ+12Ψ/N` / `2Ψ+14Ψ/N` / `16Ψ/N` — exactly, to the byte.
3. **FLOPs are invariant.** Asserted identical across stages, so a stage cannot secretly buy memory
   with recomputation.
4. **Optimizer work is sharded.** Each GPU's Adam step is asserted to touch exactly `Ψ` elements
   (baseline) or `Ψ/N` (every ZeRO stage).

```
max |loss - baseline|        ZeRO-1 (Pos) : 0.000e+00
max |loss - baseline|      ZeRO-2 (Pos+g) : 0.000e+00
max |loss - baseline|    ZeRO-3 (Pos+g+p) : 0.000e+00

            config     measured B      formula B   match
       Baseline DP      6,390,272      6,390,272   ok
      ZeRO-1 (Pos)      1,747,340      1,747,340   ok
    ZeRO-2 (Pos+g)        973,518        973,518   ok
  ZeRO-3 (Pos+g+p)        199,696        199,696   ok
```

---

## Running it

**Colab** — click the badge at the top. Nothing to install.

**Locally** — NumPy and Matplotlib are the only dependencies:

```bash
pip install numpy matplotlib
jupyter notebook ZeRO_32_GPU_Simulation.ipynb
```

Run all cells; takes about a minute on a CPU. Figures are written to `figures/`.

### Configuration

Everything is in one cell near the top:

```python
N_GPUS   = 32                                    # virtual GPUs
LAYERS   = [(256, 512), (512, 512), (512, 10)]   # MLP shape
GLOBAL_BATCH = 1024                              # -> 32 samples per GPU
STEPS    = 25
```

Change `N_GPUS` to any value that divides `GLOBAL_BATCH` and rerun. Making `LAYERS` deeper and
narrower is the interesting experiment: it shrinks ZeRO-3's transient peak toward its `16Ψ/N`
persistent floor, which is the regime real models live in.

## Repository layout

```
ZeRO_32_GPU_Simulation.ipynb   the simulation, self-contained, outputs committed
figures/                       plots written by the notebook, embedded in this README
README.md
```

## Limitations

Stated plainly so the numbers are not over-read:

- Ranks execute **sequentially**. No wall-clock speedup is measured and none is claimed. Bandwidth,
  latency, and compute/communication overlap are not modelled — a real ZeRO-3 run hides much of the
  extra `Ψ` behind compute via prefetch, so its true cost is below the +50% shown here.
- **Activation memory** is measured but not optimised. Activation checkpointing and ZeRO-R are out of
  scope; on deep models activations frequently dominate, and ZeRO does not address them.
- No CPU/NVMe offload (ZeRO-Infinity), no gradient accumulation, no tensor or pipeline parallelism.
- The model is a 3-layer MLP, so a single layer is a large fraction of it — which **exaggerates**
  ZeRO-3's transient peak relative to a realistic deep transformer.
- Ring-collective volumes are analytic `(N−1)/N` formulas applied to real payload sizes, not a
  simulation of an actual network topology.

## Reference

Rajbhandari, Rasley, Ruwase, He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter
Models.* SC'20. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
