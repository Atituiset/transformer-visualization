# Transformer Architecture Evolution Visualization — Feature Guide

> Click "❓ Guide" in the top bar to open the condensed version of this guide (shown automatically on first visit). This document is the full version. 中文版见 [docs/功能说明.md](./功能说明.md)。

## 1. What is this project

A **pure front-end, single-page** interactive demo: walk through the evolution of deep learning / LLM architectures from 1943 (MP neuron) to 2026 (the inference-efficiency race). The key point: **all forward computation is real TensorFlow.js execution**, not simulated animation — the attention heatmaps, generated text, and fine-tuning loss curves all come from real tensor math.

Tech stack: vanilla HTML/CSS/JS · TensorFlow.js 4.x (inference) · D3.js 7 (charts) · KaTeX (formulas).

## 2. The three-column layout

| Column | Name | How to use |
|--------|------|------------|
| Left | History Timeline | 55+ milestone papers grouped into four phases. Click a node → the architecture diagram highlights the related modules and an explanation drawer opens |
| Middle | Architecture + charts | Top: the current view's diagram — click any module for its explanation; bottom: attention heatmap (layer/head/post-/pre-softmax switching), Model View, Neuron View |
| Right | Run panel | Input text → tokenization → hyperparameters → ▶ run a real forward pass; the bottom console supports play/pause/single-step/jump |

## 3. How the view buttons relate (important)

The row of buttons at the top of the middle column is not a list of parallel features — it is a **timeline of architectural evolution**:

```
Foundations (1943-2016)
  └─ Prehistory RNN→2017
       └─ 2017 Original Encoder-Decoder
            ├─ Decoder-Only·Dense   ← the GPT / LLaMA / Qwen mainstream line
            ├─ Decoder-Only·MoE     ← Switch / Mixtral / DeepSeek-V3 sparsification branch
            ├─ Attention Evolution   ← MHA→MQA→GQA→MLA, FlashAttention, NSA
            ├─ Linear·SSM           ← Linear Transformer / RWKV / Mamba
            ├─ Generation Paradigms  ← autoregressive vs MTP vs diffusion LM vs speculative decoding
            ├─ Frontier              ← BLT byte-level / memory layers / TTT·Titans
            └─ Training·Alignment    ← pre-training → SFT → RLHF/DPO + scaling laws
```

**Suggested route**: `2017 Original` (understand Encoder-Decoder and attention itself) → `Decoder-Only·Dense` (see what modern LLMs changed: RMSNorm/RoPE/GQA/SwiGLU) → then branch by interest.

## 4. Core features explained

### 1. Real forward pass (right panel ▶)
Input "我 爱 自然语言处理" → character tokenization → embedding → several Attention+FFN layers → next-character probability distribution. Three engines match three views:
- **2017 Original**: full Encoder-Decoder with step-by-step Decoder generation (cross-attention visible)
- **Dense**: a modern Decoder with RMSNorm + RoPE + GQA + SwiGLU
- **MoE**: Router top-2 sparse routing + expert load stats (note: an illustrative implementation — all experts are computed and merged by gate weights; real inference only computes selected experts)

### 2. Heatmap and three views
- Heatmap: rows = query positions, columns = key positions, color = attention weight. Causal masking makes it lower-triangular.
- **Model View**: miniature overview of all N layers × H heads; click a cell to locate it in the main view.
- **Neuron View**: click a heatmap cell to break softmax(QKᵀ/√d) down into per-dimension Q·K dot products — see which dimensions drove that attention.

### 3. Making generation semantically real (🎓 Training Studio)
- Default weights are randomly initialized → gibberish output is **expected** (mechanism demo, not capability).
- "⬇ Load weights" loads a TinyShakespeare character-level model (trained in-browser via `tools/train.html`, operator-for-operator identical to the demo architecture); Dense generation becomes real English text.
- "⚡ Fine-tune" runs real Adam backprop on the loaded weights; the loss curve plots live — the variables being trained are exactly the ones visualized.

### 4. KV Cache incremental decoding
Toggle on: prefill computes the whole prompt once and caches K/V; each subsequent step forwards only 1 new token. The decode-step heatmap = new-token attention over the full cache. A full-recompute timing comparison is included (on this tiny model incremental decoding isn't faster — expected; gains grow with context length).

### 5. Sampling controls
temperature / top-k / top-p sliders. The Top-K chart shows truncation + renormalization live — watch temperature flatten the distribution and top-k/p cut the tail.

### 6. Single-step debug console
Bottom playback bar: ▶ play / ⏸ pause / step forward / back / jump. Every log step (embedding, per-layer QKV, softmax, FFN…) is clickable, jumping to an explanation drawer (formula + tensor shapes + parameter count + runnable code).

### 7. Code export
Generates runnable tfjs model-building code from the current hyperparameters, one-click copy — what you tune on the page matches the exported code exactly.

## 5. Glossary

| Term | Plain language |
|------|----------------|
| Q/K/V | Query (what I'm looking for) / Key (what I am) / Value (what I contain) |
| GQA | Groups of Q heads share one K/V head — saves memory |
| RoPE | Rotary position encoding: rotation injects position into Q/K |
| RMSNorm | LayerNorm without mean-centering (scale only) |
| SwiGLU | SiLU-gated FFN activation |
| MoE | Mixture of Experts: a router picks top-K experts per token |
| SSM/Mamba | Recurrent state replaces attention — O(n) over long sequences |
| Speculative decoding | A small draft model guesses, the big model verifies in batch |
| KV Cache | Cache historical K/V so each autoregressive step computes only the new token |

## 6. FAQ

**Q: Why is generation gibberish?**
Default weights are random init — by design. Load pretrained weights from the Training Studio for semantic output.

**Q: Why do pretrained weights invalidate after changing hyperparameters?**
They only fit the d128/h4/N3 config; other shapes don't match, so it falls back to random init.

**Q: Why isn't MoE faster?**
The illustrative implementation computes all experts and merges by gate weight for teaching purposes; real inference computes only routed experts.

**Q: Weight loading fails when double-clicking index.html?**
fetch is restricted under file:// — serve locally: `python3 -m http.server 8642`.
