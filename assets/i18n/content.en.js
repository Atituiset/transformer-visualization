(function(){
/* ============================================================
   内容语言包 · 中文（zh）
   从 index.html 抽出的全部展示型数据：模块详解 / 时间轴 / 架构视图。
   由 index.html 的语言 loader 动态加载 zh 或 en 版本。
   ============================================================ */
const MODULE_DETAILS = {
  'input': {
    title: 'Input Tokens',
    desc: 'The source sequence is tokenized, then mapped to vocabulary indices.',
    latex: ['x=[x_1,x_2,\\dots,x_n],\\; x_i\\in\\mathbb{Z},\\; x_i<|V|'],
    shapes: ['[seq_len]', '[batch, seq_len, d_model]'],
    code: "const ids = tokens.map(t => Vocab.indexOf(t));\nconst idx = tf.tensor1d(ids, 'int32');",
    params: '0 (index lookup only)'
  },
  'embed': {
    title: 'Input Embedding',
    desc: 'Vocab indices map to d_model vectors, scaled by √d_model (original-paper practice), then added to positional encoding.',
    latex: [
      '\\text{Emb}(x)=E[x]\\cdot\\sqrt{d_{model}}',
      'E\\in\\mathbb{R}^{|V|\\times d_{model}}'
    ],
    shapes: ['[batch, seq_len]', '[batch, seq_len, d_model]'],
    code: "const emb = tf.gather(E, ids)\n            .mul(Math.sqrt(dModel))\n            .reshape([1, seqLen, dModel]);",
    params: 'V × d_model'
  },
  'pe': {
    title: 'Positional Encoding',
    desc: 'Sinusoidal functions give each position a unique clock fingerprint so the model senses word order; added element-wise to embeddings.',
    latex: [
      'PE_{(pos,\\,2i)}=\\sin\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)',
      'PE_{(pos,\\,2i+1)}=\\cos\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)'
    ],
    shapes: ['[1, max_len, d_model]', 'after addition [batch, seq_len, d_model]'],
    code: "const denom = Math.pow(10000, (2 * i) / dModel);\npe[pos*dModel+i]   = Math.sin(pos / denom); // even dims\npe[pos*dModel+i+1] = Math.cos(pos / denom); // odd dims\nconst peTensor = tf.tensor3d(pe, [1, maxLen, dModel]);",
    params: '0 (fixed function, no trainable parameters)'
  },
  'enc-mha': {
    title: 'Multi-Head Self-Attention',
    desc: 'Every position acts as Query/Key/Value simultaneously, aggregating information across the sequence; h heads compute in different subspaces in parallel, then concatenate.',
    latex: [
      '\\text{Attention}(Q,K,V)=\\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
      '\\text{head}_i=\\text{Attention}(QW_i^Q,\\,KW_i^K,\\,VW_i^V)',
      '\\text{MultiHead}(Q,K,V)=\\text{Concat}(\\text{head}_1,\\dots,\\text{head}_h)\\,W^O'
    ],
    shapes: ['[batch, seq_len, d_model]', 'intermediate [batch, h, seq_len, d_k]', '[batch, seq_len, d_model]'],
    code: "const scores  = tf.matMul(Q, K, false, true).div(Math.sqrt(dk));\nconst weights = tf.softmax(scores, -1);          // [B,H,L,L]\nconst output  = tf.matMul(weights, V);           // weighted sum of V\n// multi-head: reshape([B,L,h,dh]).transpose([0,2,1,3]) splits heads; inverse merges back",
    params: '4·(d² + d)  (W^Q,W^K,W^V,W^O and biases)'
  },
  'dec-masked-mha': {
    title: 'Masked Multi-Head Attention (Decoder)',
    desc: 'The decoder sees only the generated prefix: upper-triangle positions are set to −∞ before softmax so their probability is zero — no peeking at the future.',
    latex: [
      '\\text{Attention}(Q,K,V)=\\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}+M\\right)V',
      'M_{ij}=\\begin{cases}0 & j\\le i\\\\ -\\infty & j>i\\end{cases}'
    ],
    shapes: ['[batch, Lt, d_model]', 'mask [1, 1, Lt, Lt]', '[batch, Lt, d_model]'],
    code: "// additive mask: 0 lower triangle / -1e9 elsewhere\nconst mask = tf.where(\n  tf.greater(tf.range(0,L).reshape([1,L]),\n             tf.range(0,L).reshape([L,1])),\n  tf.fill([L,L], -1e9), tf.zeros([L,L]));\nconst masked = scores.add(mask.reshape([1,1,L,L]));",
    params: '4·(d² + d)'
  },
  'dec-cross': {
    title: 'Cross-Attention',
    desc: 'The decoder\'s Encoder-Decoder attention: Q from the decoder\'s current layer, K/V from the encoder\'s final output — the sole gateway for source information (heir to the 2014 Seq2Seq+attention idea).',
    latex: [
      '\\text{head}_i=\\text{Attention}(H_{dec}W_i^Q,\\,H_{enc}W_i^K,\\,H_{enc}W_i^V)'
    ],
    shapes: ['Q [batch, Lt, d_model]', 'K/V [batch, Ls, d_model]', 'output [batch, Lt, d_model]'],
    code: "const c = crossMha.forward(hDec, encOut, encOut, null);\n// Q from decoder hidden state; K/V from encoder output (same projection family)",
    params: '4·(d² + d)'
  },
  'addnorm': {
    title: 'Residual Connection + LayerNorm (Add & Norm)',
    desc: 'Residual bypasses let gradients reach the bottom layers directly, enabling deep stacking; LayerNorm stabilizes each layer\'s activation distribution.',
    latex: [
      'h=\\text{LayerNorm}\\big(x+\\text{Sublayer}(x)\\big)',
      '\\text{LN}(x)=\\gamma\\odot\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\varepsilon}}+\\beta'
    ],
    shapes: ['[batch, seq_len, d_model]', 'shape preserved (element-wise)'],
    code: "const { mean, variance } = tf.moments(x, -1, true);\nconst y = x.sub(mean)\n           .div(variance.add(1e-6).sqrt())\n           .mul(gamma).add(beta);\n// Add & Norm：ln(x.add(sublayerOut))",
    params: '2·d per instance (γ and β); two instances per layer → 4·d'
  },
  'ffn': {
    title: 'Feed-Forward Network',
    desc: 'An independent two-layer MLP per position: expand to d_ff (512→2048), ReLU, project back to d_model. The bulk of parameters.',
    latex: [
      '\\text{FFN}(x)=\\max\\!(0,\\,xW_1+b_1)W_2+b_2'
    ],
    shapes: ['[batch, seq_len, d_model]', '[batch, seq_len, d_ff]', '[batch, seq_len, d_model]'],
    code: "const h = tf.relu(x.matMul(W1).add(b1)); // [B,L,d_ff]\nconst out = h.matMul(W2).add(b2);        // [B,L,d_model]",
    params: '2·d·d_ff + f + d'
  },
  'enc-stack': {
    title: 'Full Encoder block (x N layers)',
    desc: 'The sublayers above chain into one block, stacked N times (N=6 in the original paper). All positions fully parallel, no recurrent dependence.',
    latex: [
      'H^{(l)}=\\text{LN}\\Big(H^{(l-1)}+\\text{FFN}\\big(\\text{LN}(H^{(l-1)}+\\text{MHA}(H^{(l-1)}))\\big)\\Big)'
    ],
    shapes: ['input [batch, seq_len, d_model]', 'shape unchanged between N layers', 'output [batch, seq_len, d_model]'],
    code: "for (let l = 0; l < numLayers; l++) {\n  const r = encLayers[l].forward(x); // MHA → Add&Norm → FFN → Add&Norm\n  x = r.output;\n}",
    params: 'per layer 4·(d²+d) + 2·d·f + f + 5·d, × N layers'
  },
  'dec-stack': {
    title: 'Full Decoder block (x N layers)',
    desc: 'Each layer adds a Cross-Attention subblock over the Encoder\'s; self-attention carries a causal mask so step t depends only on steps < t.',
    latex: [
      'z=\\text{LN}_1(x+\\text{MaskedMHA}(x)),\\; z^{\\prime}=\\text{LN}_2\\big(z+\\text{CrossAttn}(z,\\,H_{enc})\\big)'
    ],
    shapes: ['target input [batch, Lt, d_model]', 'Cross-Attention K/V [batch, Ls, d_model]', 'output [batch, Lt, d_model]'],
    code: "for (const layer of decLayers) {\n  const r = layer.forward(y, encOut, causalMask(Lt));\n  y = r.output;\n}",
    params: 'per layer 8·(d²+d) + 2·d·f + f + 7·d, × N layers'
  },
  'out': {
    title: 'Output projection + Softmax',
    desc: 'The decoder\'s last-position hidden state projects to vocab dimension; softmax yields the next-token distribution (the word-by-word generation loop). Note: this demo\'s weights are randomly initialized — the distribution demonstrates mechanism, not language ability.',
    latex: [
      'P(u\\,|\\,x)=\\text{softmax}\\big(H_L W_{out}+b\\big),\\quad W_{out}\\in\\mathbb{R}^{d\\times|V|}'
    ],
    shapes: ['[batch, 1, d_model]', '[batch, |V|]', 'probabilities sum to 1'],
    code: "const last = y.slice([0, Lt-1, 0], [1, 1, dModel]);\nconst logits = last.matMul(Wout).add(bout); // [1,1,V]\nconst probs  = tf.softmax(logits, -1);",
    params: 'd × V + V'
  },

  /* ============ 前史（RNN → Transformer） ============ */
  'rnn': {
    title: 'Vanilla RNN (Elman, 1990)',
    desc: 'The most primitive sequence model: hidden state recurses through time, each step\'s output feeding the next. Theoretically remembers arbitrary history; in practice gradient products limit memory to ~10-20 steps.',
    latex: [
      'h_t=\\tanh(W_h h_{t-1}+W_x x_t+b),\\quad y_t=W_y h_t',
      '\\frac{\\partial h_t}{\\partial h_0}=\\prod_{k\\le t} W_h^{\\top}\\text{diag}(\\varphi\\prime)\\ \\text{(repeated product → vanishing/exploding)}'
    ],
    shapes: ['[batch, seq, d] unrolled step by step','step t must wait for t−1 (no parallelism)'],
    code: "h = tf.zeros([B, d]);\nfor (const x_t of xs)          // must be serial\n  h = tf.tanh(h.matMul(Wh).add(x_t.matMul(Wx)));",
    params: 'd² + d·d + d',
    refs: ['Elman 1990 · Finding Structure in Time',
           'Rumelhart et al. 1986 · Learning representations by back-propagating errors']
  },
  'lstm': {
    title: 'LSTM (1997) and GRU (2014)',
    desc: 'A conveyor belt for the RNN: cell state cₜ flows near-linearly through time while three gates (input/forget/output) control reading, writing and erasing, easing vanishing gradients. GRU (2014) merged the gates — fewer parameters, comparable quality.',
    latex: [
      'c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde{c}_t,\\quad h_t=o_t\\odot\\tanh(c_t)',
      'i_t=\\sigma(W_i[h_{t-1},x_t])\\ \\text{(forget/output gates analogous)}'
    ],
    shapes: ['[batch, seq, 4d] (four gates concatenated, computed at once)','c_t / h_t [batch, d]'],
    code: "const gates = tf.matMul(tf.concat([h, x], 1), W);  // i f o g\nconst c = f.mul(cPrev).add(i.mul(tf.tanh(g)));     // cell-state update\nconst h = o.mul(tf.tanh(c));",
    params: '4·(d·(d+e)+d) (GRU has 3 sets)',
    refs: ['Hochreiter & Schmidhuber 1997 · Long Short-Term Memory',
           'Cho et al. 2014 · Learning Phrase Representations (GRU)']
  },
  's2s': {
    title: 'Seq2Seq（2014）',
    desc: 'The Encoder-Decoder framework: the encoder compresses a variable-length source sentence into one fixed vector c, and the decoder generates the target sentence from it step by step. It established the encode-decode paradigm — but the fixed vector became a performance bottleneck for long sentences.',
    latex: [
      'c=\\text{Enc}(x_{1:n}),\\quad P(y)=\\prod_t P(y_t\\mid y_{<t},c)'
    ],
    shapes: ['source [n, d] → c [d] (information bottleneck)','target generated step by step'],
    code: "const c = encodeRNN(src).slice([srcLen-1]);  // final state = whole sentence\nlet h = c;\nfor (let t = 0; t < maxLen; t++) h = decodeRNN(h, yPrev);",
    params: 'two sets of RNN parameters',
    refs: ['Sutskever et al. 2014 · Sequence to Sequence Learning with Neural Networks']
  },
  'batt': {
    title: 'Additive Attention (Bahdanau, 2015)',
    desc: 'Breaking the fixed-vector bottleneck: each decoding step dynamically attends over all encoder states, weighting them by alignment scores α into a context vector. Visualizing the attention weights is soft alignment — the direct ancestor of 2017 self-attention (Luong proposed the multiplicative/dot-product version the same year).',
    latex: [
      'e_{ti}=a(s_{t-1},h_i),\\quad \\alpha_{ti}=\\text{softmax}_i(e_{ti})',
      'c_t=\\textstyle\\sum_i \\alpha_{ti} h_i'
    ],
    shapes: ['scores [L_t, L_s] → weights α (rows sum to 1)','context c_t [d]'],
    code: "const e = tf.tanh(sPrev.matMul(Wa).add(H.matMul(Ua)));\nconst alpha = tf.softmax(e.matMul(v), -1);   // alignment weights\nconst ctx = alpha.matMul(H);                 // weighted context",
    params: 'alignment network ~3·d²',
    refs: ['Bahdanau et al. 2015 · Neural Machine Translation by Jointly Learning to Align and Translate',
           'Luong et al. 2015 · Effective Approaches to Attention-based Neural Machine Translation']
  },
  'selfattn': {
    title: 'Self-Attention (2017, endpoint of the prehistory line)',
    desc: 'The key leap: attention is no longer just decoder-looks-at-encoder but sequence-looks-at-itself — Q, K, V share provenance. Any two positions connect in one hop (path O(1)); all positions process in one parallel matrix multiply; recurrence is abandoned entirely.',
    latex: [
      '\\text{Attention}(X,X,X)=\\text{softmax}\\!\\left(\\frac{XX^{\\top}}{\\sqrt{d}}\\right)X'
    ],
    shapes: ['[batch, L, d] → [batch, L, d] (one parallel pass)'],
    code: "const scores = X.matMul(X, false, true).div(Math.sqrt(d));\nreturn tf.softmax(scores, -1).matMul(X);",
    params: '0 (excluding projections)',
    refs: ['Vaswani et al. 2017 · Attention Is All You Need']
  },

  /* ============ Decoder-Only · Dense 现代组件 ============ */
  'rope': {
    title: 'RoPE Rotary Position Embedding (RoFormer, 2021)',
    desc: 'Encodes position as 2D-plane rotations: q and k rotate by m·θ and n·θ respectively, so their inner product depends only on relative position m−n. Absolute-position implementation with relative-position expression, and extrapolable; with YaRN/NTK interpolation it extends to million-token contexts. Used across LLaMA/Qwen/Kimi (ALiBi 2022 is a linear-bias variant).',
    latex: [
      '\\begin{pmatrix}q_m^{(2i)}\\\\ q_m^{(2i+1)}\\end{pmatrix}=\\begin{pmatrix}\\cos m\\theta_i & -\\sin m\\theta_i\\\\ \\sin m\\theta_i & \\cos m\\theta_i\\end{pmatrix}\\begin{pmatrix}q^{(2i)}\\\\ q^{(2i+1)}\\end{pmatrix}',
      '\\langle R_m q,\\,R_n k\\rangle=\\langle q,\\,R_{n-m}k\\rangle\\ \\text{(depends only on }m-n\\text{)}'
    ],
    shapes: ['[batch, seq, d_model] paired-dimension rotation','shape preserved, 0 parameters'],
    code: "const theta = Math.pow(10000, -2*i/dModel);\nconst [x, y] = [q[2*i], q[2*i+1]];\nq[2*i]   = x*Math.cos(m*theta) - y*Math.sin(m*theta);\nq[2*i+1] = x*Math.sin(m*theta) + y*Math.cos(m*theta);",
    params: '0 (fixed rotation, no trainable parameters)',
    refs: ['Su et al. 2021 · RoFormer: Enhanced Transformer with Rotary Position Embedding',
           'Press et al. 2022 · ALiBi (Train Short, Test Long)',
           'Peng et al. 2024 · YaRN (context extension)',
           'Moonshot AI 2026 · Kimi K3 (first full-stack NoPE frontier model: positions carried implicitly by causal mask + KDA state)']
  },
  'rmsnorm': {
    title: 'RMSNorm（2019）+ Pre-Norm',
    desc: 'LayerNorm minus the mean-centering: scale by root-mean-square only — less computation, comparable quality; combined with Pre-Norm (normalize before the sublayer) even hundred-layer networks train stably. GPT-2 adopted Pre-LN; LLaMA adopted Pre-RMSNorm — Post-LN from 2017 is history.',
    latex: [
      '\\text{RMSNorm}(x)=\\frac{x}{\\sqrt{\\frac{1}{d}\\textstyle\\sum_i x_i^2+\\varepsilon}}\\odot g',
      '\\text{vs LayerNorm: no mean subtraction }\\mu(x)'
    ],
    shapes: ['[batch, seq, d_model] → same shape'],
    code: "const ms = tf.mean(tf.square(x), -1, true);\nreturn x.div(ms.add(1e-6).sqrt()).mul(gamma);",
    params: 'd (scale vector g only)',
    refs: ['Zhang & Sennrich 2019 · Root Mean Square Layer Normalization',
           'Xiong et al. 2020 · On Layer Normalization in the Transformer Architecture (Pre-LN)']
  },
  'swiglu': {
    title: 'SwiGLU gated feed-forward (2020→LLaMA)',
    desc: 'The modern LLM FFN: SiLU-gated branch element-wise multiplied with a linear branch before projection; consistently beats ReLU FFN at equal parameter count. The extra matrix means d_ff is usually ~⅔·4d to keep total parameters aligned. Activation evolution: ReLU (2017) → GELU (GPT/BERT) → SiLU/SwiGLU (LLaMA onward).',
    latex: [
      '\\text{SwiGLU}(x)=\\big(\\text{SiLU}(xW_{g})\\odot xW_{u}\\big)W_{d},\\quad \\text{SiLU}(x)=x\\cdot\\sigma(x)'
    ],
    shapes: ['[B,L,d] → [B,L,d_ff] → [B,L,d]'],
    code: "const gate = tf.silu(x.matMul(Wg));\nconst up   = x.matMul(Wu);\nreturn gate.mul(up).matMul(Wd);",
    params: '3·d·d_ff',
    refs: ['Shazeer 2020 · GLU Variants Improve Transformer',
           'Touvron et al. 2023 · LLaMA (first large-scale adoption)']
  },
  'gqa': {
    title: 'GQA Grouped-Query Attention (2023)',
    desc: 'Groups of Q heads share KV heads (e.g., 32 Q heads share 8 KV groups): KV cache and decoding bandwidth drop 4x directly, quality close to MHA; can also be converted cheaply from a trained MHA checkpoint via uptraining. MQA (2019) is the extreme h_KV=1 case. Standard on Qwen2/3, LLaMA-2/3.',
    latex: [
      '\\text{head}_i=\\text{Attention}\\big(QW_i^Q,\\,KW_{g(i)}^K,\\,VW_{g(i)}^V\\big),\\quad g(i)=\\big\\lfloor i/(h_Q/h_{KV})\\big\\rfloor'
    ],
    shapes: ['Q [B, 32, L, dh]','K/V [B, 8, L, dh] → tile broadcast','KV Cache ÷ 4'],
    code: "const K8  = K.reshape([B,L,8,dh]).transpose([0,2,1,3]);\nconst K32 = tf.tile(K8, [1, 4, 1, 1]);   // 8 KV groups → 32 heads\nconst out = sdpa(Q, K32, V32, causalMask(L));",
    params: 'KV projection saves (1 − h_KV/h_Q)·2·d·d',
    refs: ['Ainslie et al. 2023 · GQA: Training Generalized Multi-Query Transformer Models',
           'Kwon et al. 2023 · vLLM/PagedAttention (cache memory management)']
  },
  'lm-head': {
    title: 'LM Head output layer',
    desc: 'The post-Final-RMSNorm hidden state is linearly projected to vocab size and softmaxed; argmax/sampling picks the next token. Modern LLMs often share weights with the input embedding (weight tying), saving V×d parameters; decoding generates token-by-token with a KV cache.',
    latex: [
      'P(u\\,|\\,x)=\\text{softmax}(h_L W_{out}^{\\top}),\\quad W_{out}=E^{\\top}\\ \\text{(optional tying)}'
    ],
    shapes: ['[B,L,d] → [B,L,|V|] → next token'],
    code: "const logits = h.matMul(E.transpose());  // weight tying\nconst next = tf.multinomial(tf.softmax(logits.slice([0,-1])), 1);",
    params: 'd×V (0 extra when tied)',
    refs: ['Press & Wolf 2017 · Using the Output Embedding to Improve Language Models']
  },

  /* ============ Attention 演进 ============ */
  'mha-ev': {
    title: 'MHA Multi-Head Attention (2017 baseline)',
    desc: 'Each Q head gets its own K/V heads: maximum expressiveness, but at inference the KV cache grows linearly with head count (2·L·h·dₕ per layer) — the bottleneck starting point for long context and large-batch serving. Every later variant answers can KV be smaller?',
    latex: ['\\text{Cache}=2\\,L\\,h\\,d_h\\ \\text{floats / layer / sequence}'],
    shapes: ['Q/K/V [B, h, L, dh] (h independent KV groups)'],
    code: "// all h KV groups enter the cache\nconst K = splitHeads(x.matMul(WK));  // [B,h,L,dh]",
    params: '4·(d²+d)',
    refs: ['Vaswani et al. 2017 · Attention Is All You Need']
  },
  'mqa': {
    title: 'MQA Multi-Query Attention (2019)',
    desc: 'All Q heads share one K/V group: KV cache drops h-fold and decode throughput rises sharply, but expressiveness suffers notably. Recorded in the lineage as GQA\'s extreme case (h_KV=1).',
    latex: ['h_{KV}=1:\\; \\text{Cache} \\div h'],
    shapes: ['Q [B,h,L,dh]','K/V [B,1,L,dh] (shared across all heads)'],
    code: "const K1 = K.reshape([B,L,1,dh]);\nconst Kt = tf.tile(K1, [1, h, 1, 1]);",
    params: 'KV projection only 2·d·dh',
    refs: ['Shazeer 2019 · Fast Transformer Decoding: One Write-Head is All You Need']
  },
  'mla': {
    title: 'MLA Multi-head Latent Attention (DeepSeek-V2/V3, Kimi K2)',
    desc: 'Low-rank compresses KV into a latent vector c (d_c ≪ d·h); at inference the cache stores only c, restoring K/V on demand via absorption matrices: tens-of-times memory savings with no quality loss; paired with MoE it powers efficient inference for 671B/1T-class models.',
    latex: [
      'c_{KV}=W_{DKV}x\\in\\mathbb{R}^{d_c},\\quad K=W_{UK}c_{KV},\\; V=W_{UV}c_{KV}',
      '\\text{Cache}: 2Lhd_h \\rightarrow L d_c\\ (\\div \\text{tens of times})'
    ],
    shapes: ['x [B,L,d] → c_KV [B,L,d_c≈512]','K/V restored on demand [B,h,L,dh]'],
    code: "const c = x.matMul(W_DKV);              // the only thing entering the cache\nconst K = c.matMul(W_UK), V = c.matMul(W_UV); // decode-side up-projection",
    params: 'compression/up-projection ~2·d·d_c + 2·d_c·d',
    refs: ['DeepSeek-AI 2024 · DeepSeek-V2 Technical Report (MLA)',
           'DeepSeek-AI 2024 · DeepSeek-V3 Technical Report',
           'Moonshot AI 2025 · Kimi K2 (keeps MLA)']
  },
  'flash': {
    title: 'FlashAttention (2022, IO-aware exact attention)',
    desc: 'Changes memory access, not math: Q/K/V tiles live in SRAM with online softmax per tile (maintaining running max m and sum ℓ); the L×L matrix never hits HBM. Long-context training speeds up 2-4x, memory O(L²)→O(L); the default kernel of modern frameworks (v2/2023 re-tiled the parallel axis, v3/2024 added FP8).',
    latex: [
      'm\\leftarrow\\max(m,\\,m_{\\text{tile}}),\\quad \\ell\\leftarrow e^{m_{old}-m}\\,\\ell+\\textstyle\\sum_{\\text{tile}}e^{s-m}'
    ],
    shapes: ['HBM traffic: O(L²) → O(Ld)','results identical to standard softmax'],
    code: "for (const [Kj, Vj] of tiles(K, V)) {   // K/V tiles stream in\n  const S  = Qtile.matMul(Kj.T).div(Math.sqrt(dk));\n  mNew = rowmax(m, S); l = exp(m−mNew)*l + rowsum(exp(S−mNew));\n  acc  = exp(m−mNew)*acc + exp(S−mNew).matMul(Vj); m = mNew;\n}\nconst O = acc.div(l);",
    params: '0 (kernel-level optimization, no new parameters)',
    refs: ['Dao et al. 2022 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
           'Dao 2023 · FlashAttention-2', 'Shah et al. 2024 · FlashAttention-3'],
    extra: '<svg width="330" height="168" role="img" aria-label="FlashAttention tiled online softmax illustration" style="background:var(--c-code-bg);border-radius:8px;padding:6px;">' +
      '<text x="14" y="18" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">Q×Kᵀ tiles (causal)</text>' +
      (() => { let s=''; for(let i=0;i<4;i++)for(let j=0;j<4;j++){ const on=j<=i;
        s+='<rect x="'+(14+j*26)+'" y="'+(26+i*22)+'" width="22" height="18" rx="3" fill="'+
        (on?'rgba(126,166,255,.75)':'rgba(160,175,200,.18)')+'"/>'; }
        s+='<rect x="12" y="24" width="106" height="90" rx="6" fill="none" stroke="var(--c-accent)" stroke-width="1.6" stroke-dasharray="4 3"/>';
        return s; })() +
      '<text x="140" y="40" font-size="12" fill="var(--c-text)" font-family="var(--font-mono)">SRAM-resident:</text>' +
      '<text x="140" y="58" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">Q tile + m, ℓ, O accumulators</text>' +
      '<text x="140" y="84" font-size="12" fill="var(--c-text)" font-family="var(--font-mono)">HBM：</text>' +
      '<text x="140" y="102" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">the L×L matrix is never materialized</text>' +
      '<path d="M 14 132 L 118 132" stroke="var(--c-accent)" stroke-width="1.6" marker-end="url(#arrow)"/>' +
      '<text x="14" y="152" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">K/V tiles stream in → online softmax correction</text>' +
      '</svg>'
  },
  'sparse': {
    title: 'Trainable sparse attention (NSA / MoBA, 2025)',
    desc: 'Most of the attention matrix is low-value noise: compute exact attention only on important blocks, skipping or compressing the rest. DeepSeek NSA bakes sparse patterns into pre-training (hardware-aligned block-sparse kernels); Kimi MoBA mixes full/sparse attention via block-level top-K gating, near-linear long-context cost.',
    latex: [
      'O=\\text{Attn}(Q,\\,\\text{TopK-blocks}(K,V)),\\quad g=\\text{softmax}(\\text{gate})'
    ],
    shapes: ['L×L → keep only ~k blocks per row','complexity ~O(L·k·w)'],
    code: "// MoBA: block-level gating picks top-K blocks\nconst blockScores = qBlock.matMul(kBlocks.T);\nconst topK = tf.topk(blockScores, k);  // other blocks not computed",
    params: '0 to a few gating parameters',
    refs: ['DeepSeek-AI 2025 · Native Sparse Attention (NSA)',
           'Moonshot AI 2025 · MoBA: Mixture of Block Attention for Long-Context LLMs',
           'DeepSeek-AI 2025 · V3.2-Exp (DSA sparse attention, NSA engineering)']
  },
  'hybrid': {
    title: 'Hybrid architecture (few full-attention + many linear layers)',
    desc: 'Full attention handles precise retrieval; linear layers handle cheap sequence modeling, mixed proportionally (often 1:3 to 1:7): Jamba (AI21, Mamba+Attention), Griffin/Hawk (DeepMind, gated linear recurrence), Samba (Microsoft), Kimi Linear (KDA+MLA) — a new balance of long-context efficiency and quality.',
    latex: ['\\text{layer stack}=[\\underbrace{\\text{Linear}}_{3/4},\\dots,\\underbrace{\\text{Full}}_{1/4}]\\times N'],
    shapes: ['same Decoder block, swapping attention type per layer'],
    code: "for (l of layers) x = l.isFull ? fullAttn(x) : linearAttn(x);",
    params: 'between Dense and Linear',
    refs: ['Lieber et al. 2024 · Jamba: A Hybrid Transformer-Mamba Language Model',
           'De et al. 2024 · Griffin: Mixing Gated Linear Recurrences with Local Attention',
           'Moonshot AI 2025 · Kimi Linear']
  },

  /* ============ 线性 / SSM / RNN 复兴 ============ */
  'linear': {
    title: 'Linear attention (Katharopoulos, 2020)',
    desc: 'Rewrites softmax attention with a feature map φ using associativity: compute φ(K)ᵀV first (a d×d matrix), dropping complexity from L² to O(L·d²); the same formula rewrites as state recursion Sₜ=Sₜ₋₁+φ(kₜ)vₜᵀ — Transformers are RNNs. RWKV/xLSTM/KDA later added gating and decay improvements along this line.',
    latex: [
      '\\text{Attn}(Q,K,V)=\\frac{\\phi(Q)\\big(\\phi(K)^{\\top}V\\big)}{\\phi(Q)\\phi(K)^{\\top}\\mathbf{1}}',
      'S_t=S_{t-1}+\\phi(k_t)v_t^{\\top},\\quad u_t=\\frac{S_t\\phi(q_t)}{\\phi(q_t)^{\\top}z_t}'
    ],
    shapes: ['φ(K)ᵀV: [d,d] constant state','O(d²) per step, independent of L'],
    code: "const KV = phi(K).transpose().matMul(V);  // [d,d] computed first\nconst out = phi(Q).matMul(KV);\n// streaming inference: S = S.add(phi(k_t).outer(v_t));",
    params: '0 (φ is a fixed feature map)',
    refs: ['Katharopoulos et al. 2020 · Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention']
  },
  'rwkv': {
    title: 'RWKV（2022）',
    desc: 'Rewrites attention as a linear RNN with channel-wise time decay: parallelizable training (Transformer-like), O(1) inference state (RNN-like). An evergreen linear-model line in the open-source community.',
    latex: [
      'wkv_t=\\frac{\\sum_i e^{-(t-1-i)w+k_i}v_i+e^{u+k_t}v_t}{\\sum_i e^{-(t-1-i)w+k_i}+e^{u+k_t}}'
    ],
    shapes: ['scalar decay w per channel','[d]-scale state'],
    code: "// channel-wise decay recursion (numerically stable version omitted)\na = a.mul(wDecay).add(v_t); b = b.mul(wDecay).add(1);",
    params: '~ a Transformer block',
    refs: ['Peng et al. 2023 · RWKV: Reinventing RNNs for the Transformer Era']
  },
  'xlstm': {
    title: 'xLSTM (2024, LSTM revival)',
    desc: 'The LSTM originators\' answer 27 years later: sLSTM uses exponential gating + new memory mixing; mLSTM uses matrix memory + covariance update (parallelizable). Part of the RNN-revival wave alongside Mamba/RWKV.',
    latex: ['c_t=f_t\\odot c_{t-1}+i_t\\,v_t,\\quad C_t=C_{t-1}\\odot f_t+n_t v_t^{\\top}\\ \\text{(mLSTM matrix memory)}'],
    shapes: ['memory expanded from vector [d] to matrix [d,d]'],
    code: "C = C.mul(f).add(n.outer(v));  // matrix cell state\nh = C.matMul(q).div(norm);",
    params: '~1.5× LSTM',
    refs: ['Beck et al. 2024 · xLSTM: Extended Long Short-Term Memory']
  },
  's4': {
    title: 'S4 Structured State Space (2021)',
    desc: 'Continuous state-space systems, with HiPPO initialization + structured diagonalization, become long convolutions for parallel training and recurrence for inference: swept the LRA long-sequence benchmarks — Mamba\'s direct predecessor.',
    latex: [
      "h'(t)=\\bar{A}h(t)+\\bar{B}x(t),\\quad y=\\bar{C}h(t)\\ \\text{(discretized SSM)}"
    ],
    shapes: ['kernel K = CĀᵏB̄: a length-L convolution kernel'],
    code: "const K = ssmConvKernel(A, B, C, L);  // long-convolution form\nconst y = conv1d(x, K);               // parallel training\n// inference: h = A.mul(h).add(B.mul(x));",
    params: 'structured A/B/C parameters ~O(d·N)',
    refs: ['Gu et al. 2021 · Efficiently Modeling Long Sequences with Structured State Spaces (S4)']
  },
  'mamba': {
    title: 'Mamba selective SSM (2023)',
    desc: 'Makes SSM Δ/B/C input-dependent (the selection mechanism): the model can forget or retain by content, fixing SSM\'s one-size-fits-all flaw. With hardware-aware parallel scans, training is 5x faster and inference keeps O(1) state. Mamba-2 (2024) proved its SSD unification with linear attention.',
    latex: [
      'h_t=\\bar{A}(x_t)h_{t-1}+\\bar{B}(x_t)x_t,\\quad y_t=\\bar{C}(x_t)h_t\\ \\text{(parameters vary with input)}'
    ],
    shapes: ['state [B,d,N]','scan O(L·d·N)'],
    code: "for (const x_t of xs) {            // hardware-aware scan (parallel prefix sum)\n  h = Abar(x_t).mul(h).add(Bbar(x_t).mul(x_t));\n  y_t = Cbar(x_t).dot(h);\n}",
    params: 'Δ/B/C projections ~3·d·d',
    refs: ['Gu & Dao 2023 · Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
           'Dao & Gu 2024 · Transformers are SSMs (Mamba-2/SSD)']
  },
  'kda': {
    title: 'Kimi Linear · KDA（2025）',
    desc: 'Moonshot\'s hybrid linear architecture: KDA (Kimi Delta Attention — linear attention with gated incremental memory enabling fine-grained forgetting) makes up 3/4 of layers, MLA full attention 1/4. At 3B/48B-A3B scale it matches same-size full-attention models across the board: million-token context, ~75% memory savings, ~6x decode throughput.',
    latex: [
      'S_t=\\alpha_t\\odot S_{t-1}+\\beta_t\\,\\phi(k_t)v_t^{\\top}\\ \\text{(gated delta rule)}'
    ],
    shapes: ['[d,d] matrix memory state','layer ratio Linear : Full = 3 : 1'],
    code: "for (l of layers)\n  x = (l % 4 === 0) ? mlaBlock(x) : kdaBlock(x);  // 1/4 full attention",
    params: '~ Dense Block (linear layers save more KV)',
    refs: ['Moonshot AI 2025 · Kimi Linear: An Expressive, Efficient Attention Architecture']
  },

  /* ============ MoE ============ */
  'moe-router': {
    title: 'MoE gating router (Router / Gate)',
    desc: 'A router scores every token and activates only Top-K experts (often K=8): total parameters decouple from per-token compute. Lineage: GShard(2020) → Switch(2021, Top-1) → Mixtral(2023, Top-2) → DeepSeek-V3(256 choose 8 + aux-loss-free balancing + shared expert) → Qwen3(128 choose 8) → Kimi K2(384 choose 8).',
    latex: [
      'g=\\text{softmax}\\big(\\text{TopK}(xW_g,\\,K)\\big),\\quad y=\\sum_{i\\in\\text{TopK}} g_i\\,E_i(x)',
      '\\text{DeepSeek-V3: bias-based dynamic balancing replaces the auxiliary loss (}\\mathcal{L}_{aux}\\text{-free balancing)}'
    ],
    shapes: ['x [B,L,d] → logits [B,L,E]','Top-K mask → only K/E experts computed'],
    code: "const logits = x.matMul(Wg);                 // [B,L,E]\nconst {values, indices} = tf.topk(logits, K);\nconst gate = tf.softmax(values, -1);          // sparse gating\nlet y = tf.zerosLike(x);\nfor (let k = 0; k < K; k++)\n  y = y.add(experts[indices[k]](x).mul(gate.slice(k)));",
    params: 'd×E + E (routing matrix)',
    refs: ['Shazeer et al. 2017 · Outrageously Large Neural Networks (Sparsely-Gated MoE)',
           'Lepikhin et al. 2020 · GShard', 'Fedus et al. 2021 · Switch Transformers',
           'DeepSeek-AI 2024 · Auxiliary-Loss-Free Load Balancing']
  },
  'moe-expert': {
    title: 'MoE expert (Expert = SwiGLU FFN)',
    desc: 'Each expert is just a SwiGLU FFN. Fine-grained experts (more, smaller ones) + shared experts (always-on general knowledge; adopted by DeepSeek/Qwen3) are today\'s mainstream; K2 goes further with MLA attention + 384 experts for 1T total / 32B active params.',
    latex: ['E_i(x)=\\text{SwiGLU}_i(x),\\quad \\text{active params} \\approx K/E'],
    shapes: ['[B,L,d] → [B,L,d_ff] → [B,L,d] (routed tokens only)'],
    code: "class Expert {  // = SwiGLU FFN\n  forward(x){ return tf.silu(x.matMul(Wg)).mul(x.matMul(Wu)).matMul(Wd); }\n}",
    params: 'per expert 3·d·d_ff (total params ×E, active ×K/E)',
    refs: ['Jiang et al. 2024 · Mixtral of Experts',
           'Moonshot AI 2025 · Kimi K2: Open Agentic Intelligence']
  },
  'moe-sum': {
    title: 'Sparse weighted synthesis',
    desc: 'Only the K selected experts\' outputs join the synthesis; unselected experts compute nothing this step — the essence of trillion-total/billion-active sparsity. At inference, experts can be distributed across GPUs (expert parallelism).',
    latex: ['y=\\sum_{i\\in\\text{TopK}} g_i E_i(x)'],
    shapes: ['[B,L,d] (same shape as Dense FFN output; direct residual)'],
    code: "y = g1*E1(x) + g2*E2(x);  // remaining E-2 experts skipped",
    params: '0',
    refs: ['Fedus et al. 2022 · A Review of Sparse Expert Models in Deep Learning']
  },

  /* ============ Attention 路线 C：长上下文与稳定化 ============ */
  'swa': {
    title: 'Sliding-window attention (SWA) and local/global interleaving',
    desc: 'Each position sees only the last w tokens: L×L → L×w and inference cache fixed at window size. Pioneered by Mistral; Gemma 2/3 interleave local:global ≈ 5:1 — local layers handle nearby detail while sparse global layers handle long-range retrieval. The first-token attention sink phenomenon is explained and exploited by attention sinks (StreamingLLM).',
    latex: [
      'A_{ij}=\\text{softmax}\\!\\left(\\frac{q_i k_j^{\\top}}{\\sqrt{d}}\\right),\\quad j\\in(i-w,\\,i]',
      '\\text{Cache}: O(L)\\rightarrow O(w),\\quad \\text{layer ratio local:global}=5{:}1\\ \\text{(Gemma 3)}'
    ],
    shapes: ['attention matrix [L,L] → banded [L,w]','KV cache fixed at w'],
    code: "const start = Math.max(0, t - W);\nconst s = q[t].matMul(K.slice([start, t-start+W]), false, true);\n// softmax within window only; outside is not computed",
    params: '0',
    refs: ['Jiang et al. 2023 · Mistral 7B', 'Riviere et al. 2024 · Gemma 2',
           'Google 2025 · Gemma 3 Technical Report',
           'Xiao et al. 2023 · Efficient Streaming Language Models (Attention Sinks)']
  },
  'ctx-ext': {
    title: 'Context extension: PI → NTK-aware → YaRN',
    desc: 'Three generations of stretching RoPE models from 4K training to million-token inference: PI linear interpolation (positions ÷s), simple but damages high frequencies; NTK-aware extrapolates high frequencies and interpolates low ones; YaRN adds wavelength-wise frequency-band interpolation + attention temperature compensation — a 10x extension needs only ~1000 fine-tuning steps. The technology under CodeLlama-100K and Qwen-1M.',
    latex: [
      '\\text{PI}: \\theta_m\\rightarrow\\theta_m/s;\\quad \\text{YaRN}: \\text{wavelength-wise } r(\\theta) \\text{ frequency-band interpolation} + \\sqrt{1/t}\\ \\text{temperature}'
    ],
    shapes: ['effective context ×10~250 (4K→1M)','fine-tuning cost ~1000 steps'],
    code: "// YaRN: wavelength-based blended interpolation for frequency i\nconst r = 2*Math.PI/theta[i];          // wavelength\nconst scale = r < base ? 1 : Math.max(0, (r-base)/(L_train-base)) * (1-1/s) + 1/s;",
    params: '0',
    refs: ['Chen et al. 2023 · Extending Context Window via Position Interpolation',
           'bloc97 2023 · NTK-Aware Scaled RoPE', 'Peng et al. 2024 · YaRN']
  },
  'qknorm': {
    title: 'Training stabilization trio: QK-Norm · soft-capping · z-loss',
    desc: 'Mainstream countermeasures as scale makes training brittle: QK-Norm RMS-normalizes Q/K per head to prevent attention-logit explosions (Gemma 3, Chameleon); logit soft-capping bounds outputs with tanh to prevent overconfidence (Gemma 2); z-loss penalizes softmax-denominator drift (ST-MoE/PaLM). Same stabilization-evolution line as Pre-Norm/RMSNorm.',
    latex: [
      '\\hat{q}=\\text{RMSNorm}(q)\\cdot\\sqrt{d_h};\\quad \\hat{\\ell}=c\\cdot\\tanh(\\ell/c)\\ \\text{(soft-cap)}',
      '\\mathcal{L}_z=\\frac{1}{B}\\sum_b\\big(\\log Z_b\\big)^2\\ \\text{(z-loss)}'
    ],
    shapes: ['attention logit variance kept under control','training loss curve smooths out'],
    code: "const qn = rmsNorm(Q).mul(Math.sqrt(dh));\nconst kn = rmsNorm(K).mul(Math.sqrt(dh));\nconst logits = softCap(qn.matMul(kn.T), 30); // tanh soft-cap with c=30",
    params: 'QK-Norm: 4·dₕ per head',
    refs: ['Riviere et al. 2024 · Gemma 2', 'Google 2025 · Gemma 3',
           'Chameleon Team 2024 · Chameleon Mixed-Modal Early-Fusion',
           'Zoph et al. 2022 · ST-MoE (z-loss)']
  },
  'diff': {
    title: 'Differential attention (DIFF Transformer, 2024)',
    desc: 'Modeled on a differential amplifier: two independent softmax attentions subtract, canceling common-mode noise (uniform attention to irrelevant tokens) — hallucination-resistant, long-context robust, and saves half the heads (h/2 per group). Validated at 65B scale, beating same-size Transformers.',
    latex: [
      '\\text{DiffAttn}(Q)=\\big(\\text{softmax}(Q_1K_1^{\\top})V_1-\\lambda\\,\\text{softmax}(Q_2K_2^{\\top})V_2\\big),\\ \\lambda\\in(0,1)'
    ],
    shapes: ['h/2 heads per group × 2 groups','output same shape as MHA'],
    code: "const A1 = softmax(Q1.matMul(K1.T)); const A2 = softmax(Q2.matMul(K2.T));\nreturn A1.sub(A2.mul(lambda)).matMul(V);",
    params: '~ MHA (two half-width projection sets)',
    refs: ['Ye et al. 2024 · Differential Transformer']
  },

  /* ============ 生成范式 ============ */
  'autoreg': {
    title: 'Autoregressive next-token prediction (NTP)',
    desc: 'The first principle of LLMs: causal masking + cross-entropy, left-to-right token by token. The entire GPT family (GPT/Qwen/LLaMA/Kimi/DeepSeek) uses it; KV cache makes each generation step O(L). Understanding why autoregression wins = understanding a simple objective plus emergence at scale.',
    latex: [
      'P(y)=\\prod_t P(y_t\\mid y_{<t}),\\quad \\mathcal{L}=-\\sum_t\\log P(y_t\\mid y_{<t})'
    ],
    shapes: ['logits [B,L,|V|]','O(L) per generation step (with cache)'],
    code: "const logits = model(yPrev);\nconst loss = tf.losses.softmaxCrossEntropy(oneHot(yNext), logits);",
    params: '0 (objective function, not a parameter)',
    refs: ['Radford et al. 2018/2019 · GPT-1/2', 'Brown et al. 2020 · GPT-3']
  },
  'mtp': {
    title: 'Multi-token prediction MTP (DeepSeek-V3)',
    desc: 'Beyond next-token prediction, attach D lightweight heads predicting future tokens 1..D at once: denser training signal forces the model to plan; V3 measured +1.5~3% across benchmarks. At inference these heads double as speculative-decoding drafts (the EAGLE idea) — two birds with one stone.',
    latex: [
      'p_t^{(k)}=\\text{head}_k(h_t),\\quad \\mathcal{L}=\\sum_{k=1}^{D}\\mathbb{E}\\big[-\\log p_{t+k}^{(k)}\\big]'
    ],
    shapes: ['shared trunk + D projection heads','per head [B,L,|V|]'],
    code: "for (let k = 1; k <= D; k++)\n  loss += ce(heads[k](h.slice([0, 0], [-1, L-k])), y.slice([0, k]));",
    params: 'D·d·|V| (auxiliary heads)',
    refs: ['DeepSeek-AI 2024 · DeepSeek-V3 Technical Report (MTP)',
           'Gloeckle et al. 2024 · Better & Faster Large Language Models via Multi-token Prediction',
           'Li et al. 2024 · EAGLE-2']
  },
  'diffusion': {
    title: 'Diffusion language models (LLaDA · Mercury · Gemini Diffusion)',
    desc: 'Generation = iterative denoising from full [MASK]: each round predicts a subset of positions in parallel, breaking left-to-right dependence — naturally parallel, revisable and controllably editable. LLaDA 8B first validated diffusion LM scaling laws (comparable to LLaMA3-8B); Mercury/Gemini Diffusion emphasize 5-10x generation speed. Current weaknesses: variable-length sequences, KV cache inapplicability, and top-tier quality still belongs to autoregression.',
    latex: [
      'x_T=\\text{[MASK]}^L\\ \\xrightarrow{\\ p_\\theta(y_t\\mid x_t)\\ }\\ x_0=y\\quad\\text{(per-position parallel denoising)}'
    ],
    shapes: ['all positions predicted in parallel each step','generation speed 5-10x AR'],
    code: "let x = maskAll(L);\nfor (let step = T; step > 0; step--)\n  x = denoise(x, model, step);   // reveal part of the tokens each step",
    params: '~ same-size AR model',
    refs: ['Nie et al. 2025 · LLaDA: Large Language Diffusion Models',
           'Inception Labs 2025 · Mercury Diffusion LLM',
           'Google DeepMind 2025 · Gemini Diffusion']
  },
  'spec': {
    title: 'Speculative decoding (Speculative Decoding · EAGLE)',
    desc: 'Inference acceleration rather than a new model: a small draft model guesses k tokens at once and the big model verifies them in one parallel forward pass; rejection sampling guarantees the output distribution exactly matches token-by-token generation — a math-lossless 2-4x speedup. EAGLE-style drafts use the target model\'s feature layer + MTP-like heads, now the industrial default. Note the boundary: speculative decoding changes no structure/parameters/objective — it belongs to the inference-system layer alongside KV-cache management (PagedAttention), continuous batching and quantization. Architecture decides what one forward pass computes; this decides how many forward passes to call.',
    latex: [
      '\\text{acceptance rate }\\alpha=\\mathbb{E}\\big[\\min(1,p/q)\\big],\\quad \\text{expected speedup}\\approx\\frac{1-\\alpha^{k+1}}{(1-\\alpha)}\\ \\text{x}'
    ],
    shapes: ['draft k tokens → one parallel verification','≥1 token produced per step','zero changes to the model itself'],
    code: "const draft = smallModel.kTokens(prefix, k);   // guess k tokens\nconst p = targetModel.probs(prefix.concat(draft));\nconst out = rejectSample(draft, p);            // lossless verification",
    params: 'draft model ~1/10 parameters (system-level overhead)',
    refs: ['Leviathan et al. 2023 · Fast Inference from Transformers via Speculative Decoding',
           'Chen et al. 2023 · Accelerating LLM Decoding with Speculative Sampling',
           'Li et al. 2024 · EAGLE-2',
           'Kwon et al. 2023 · vLLM/PagedAttention (also inference-system layer)',
           'Yu et al. 2022 · Orca (continuous batching, also inference-system layer)']
  },

  /* ============ 前沿探索 ============ */
  'blt': {
    title: 'Byte Latent Transformer（BLT, Meta 2024）',
    desc: 'Drops the BPE tokenizer: byte streams dynamically cluster into patches by the entropy of the next byte — the more predictable, the larger the patch; compute is allocated by information entropy. At equal training compute it beats Llama 3 on robustness (noise-resistant, no OOV, fair across languages); the cost is that long patches rely on a small local byte model. Same compute-allocation efficiency line as MTP/SWA.',
    latex: [
      '\\text{patch boundaries} \approx \\text{entropy-rate jumps},\\quad \\text{compute} \propto \\text{entropy}'
    ],
    shapes: ['bytes [B] → dynamic patches [P] → global hidden states','P varies with content'],
    code: "const patches = entropyPatcher(bytes, smallModel); // dynamic boundaries\nconst h = globalModel(patches);\nconst bytesOut = localDecoder(h, bytes);",
    params: 'patcher (small) + global model + local decoder',
    refs: ['Pagnoni et al. 2024 · Byte Latent Transformer: Patches Scale Better Than Tokens']
  },
  'memlayer': {
    title: 'Memory Layers at Scale (Meta 2024)',
    desc: 'Replace part of the FFN with trainable sparse key-value memory (million-scale embeddings, product-key top-k retrieval): large factuality gains at tiny compute cost — an engineering validation of the division-of-labor hypothesis store knowledge in memory, reason via attention. 128M memory gave +100%+ trivia gains on a 1.3B model.',
    latex: [
      'y=\\textstyle\\sum_{i\\in\\text{topk}(qK)} v_i\\quad\\text{(product-key quantized retrieval)}'
    ],
    shapes: ['query [d] → top-k memory slots','sparse activation ~k per million'],
    code: "const keys1 = q.matMul(PK1); keys2 = q.matMul(PK2);\nconst topk = topkProduct(keys1, keys2, k);   // factorized retrieval\nconst y = V.gather(topk).mean(0);",
    params: 'memory table 1M~1B×d (sparse access)',
    refs: ['Berges et al. 2024 · Memory Layers at Scale (Meta)']
  },
  'ttt': {
    title: 'TTT layer: test-time training (2024)',
    desc: 'Replaces the RNN hidden state with an online-learnable neural network: each arriving token takes one gradient step on the inner model; the output is the updated model\'s forward pass. Sequence modeling becomes test-time training — state capacity upgrades from a fixed vector to an entire network.',
    latex: [
      'W_t=W_{t-1}-\\eta\\,\\nabla\\,\\ell\\big(W_{t-1};x_t\\big),\\quad y_t=f\\big(W_t;x_t\\big)'
    ],
    shapes: ['state = weights W [parameter-scale]','one SGD step per token'],
    code: "for (const x_t of xs) {\n  const g = grad(w => reconLoss(w, x_t));   // self-supervised\n  W = W.sub(g.mul(eta));                    // state update\n  y_t = f(W, x_t);\n}",
    params: 'inner model ~MLP-scale',
    refs: ['Sun et al. 2024 · Learning to (Learn at Test Time): RNNs with Expressive Hidden States']
  },
  'titans': {
    title: 'Titans: test-time neural memory (Google 2025)',
    desc: 'Surprise-driven memory: only information that contradicts current memory predictions gets written into the long-term neural memory module (MLP-scale, extensible to tens-of-millions context), alongside short-term attention and persistent memory. The engineering exemplar of the TTT/Memorizing Transformer line: memory keeps learning at test time.',
    latex: [
      'M_t=M_{t-1}-\\theta_t\\,\\nabla\\,\\ell\\big(M_{t-1};x_t\\big)\\quad\\text{(surprise = gradient magnitude)}'
    ],
    shapes: ['long-term memory at [d,d] scale','2M→10M+ context'],
    code: "const surprise = grad(M => reconLoss(M, x_t));\nM = M.mul(forgetGate).sub(surprise.mul(theta));\ny = attn(x_short) + M.query(q_t);",
    params: 'memory module ~MLP-scale',
    refs: ['Behrouz et al. 2025 · Titans: Learning to Memorize at Test Time',
           'Wu et al. 2022 · Memorizing Transformers']
  },
  'looped': {
    title: 'Deep looping / latent reasoning (Universal Transformer → 2025)',
    desc: 'The same layers iterate repeatedly with depth adapting to problem difficulty: easy problems loop less, hard ones think longer — moving test-time compute from output tokens to latent-space recursion. The 2018 Universal Transformer returned in 2025 as latent-reasoning models (e.g. Huginnet), complementing o1-style long chains of thought.',
    latex: [
      'h^{(k+1)}=\\text{Block}\\big(h^{(k)}\\big),\\quad K\\ \\text{adaptive by difficulty (AdaTape/ACT)}'
    ],
    shapes: ['parameter count unchanged','effective depth K×N'],
    code: "let h = embed(x);\nfor (let k = 0; k < adaptiveSteps(difficulty); k++)\n  h = block(h);   // recursive depth with shared weights",
    params: 'N layers of parameters (shared, reused K times)',
    refs: ['Dehghani et al. 2018 · Universal Transformers',
           'Geiping et al. 2025 · Reasoning by Latent Space (Huginnet)']
  },
  'multimodal': {
    title: 'Three routes of multimodal fusion',
    desc: '1) Adapters: ViT/Whisper encoder outputs join the LLM via a projector (LLaVA, Qwen-VL) - the mainstream; 2) early fusion: image-text tokens mixed in unified training (Chameleon); 3) natively omni-modal: end-to-end streaming input/output (GPT-4o family). The shared idea: turn every modality into token sequences for the same Transformer.',
    latex: [
      'h=\\text{LLM}\\big(\\big[\\text{Proj}(v_1..v_m);\\;t_1..t_n\\big]\\big)'
    ],
    shapes: ['image → ViT patch tokens → projected to d_model','concatenated with text tokens'],
    code: "const vis = proj(vitEncoder(image));       // [m, d]\nconst h = llm(tf.concat([vis, textTokens], 1));",
    params: 'ViT + projector (~5-15% of the LLM)',
    refs: ['Liu et al. 2023 · Visual Instruction Tuning (LLaVA)',
           'Chameleon Team 2024 · Mixed-Modal Early-Fusion Foundation Models',
           'Bai et al. 2023 · Qwen-VL']
  },

  /* ============ 训练与对齐 ============ */
  'pretrain': {
    title: 'Pre-training',
    desc: 'Stage one of the LLM three-stage pipeline: self-supervised learning over massive unlabeled text — decoder models use next-token prediction (NTP), BERT uses masked language modeling (MLM), T5 uses span corruption. Language, knowledge and basic reasoning all come from here; later alignment only shapes behavior and adds almost no new capability (hence the alignment-tax debate).',
    latex: [
      'L_{\\text{NTP}}=-\\sum_t \\log P(y_t\\mid y_{<t}),\\quad L_{\\text{MLM}}=-\\sum_{t\\in M}\\log P(x_t\\mid x_{\\setminus M})'
    ],
    shapes: ['corpus ~10T tokens → batches [B, L]','compute on the order of ~10^25 FLOPs'],
    code: "// mix NTP (Decoder) / MLM (Encoder) / span corruption (T5)\nconst loss = tf.losses.softmaxCrossEntropy(oneHot(next), logits);",
    params: 'all model parameters are learned at this stage',
    refs: ['Radford et al. 2018/2019 · GPT series', 'Devlin et al. 2018 · BERT',
           'Raffel et al. 2019 · T5']
  },
  'sft': {
    title: 'SFT supervised fine-tuning (Supervised Fine-Tuning)',
    desc: 'Stage two: supervised fine-tuning on human-written instruction→response demonstration pairs (same loss as pre-training, new data). Turns the base model from continuer into instruction-follower. FLAN/T0 showed instruction generalization transfers to unseen tasks; LIMA showed a well-aligned base needs only ~1000 high-quality demonstrations — data quality far outweighs quantity.',
    latex: [
      'L_{\\text{SFT}}=-\\sum_t \\log p(y_t\\mid \\text{instruction},\\, y_{<t})'
    ],
    shapes: ['demonstration pairs (instruction, response) ~1k to 1M','same structure as pre-training, data swapped'],
    code: "// same loss as pre-training; data swapped to instruction demonstrations\nfor (const {prompt, answer} of sftData) loss += ce(model(prompt), answer);",
    params: 'full fine-tuning or LoRA (low-rank bypass)',
    refs: ['Wei et al. 2021 · Finetuned Language Models are Zero-Shot Learners (FLAN)',
           'Sanh et al. 2021 · Multitask Prompted Training (T0)',
           'Ouyang et al. 2022 · InstructGPT',
           'Zhou et al. 2023 · LIMA: Less Is More for Alignment']
  },
  'rlhf': {
    title: 'RLHF: RL from human feedback',
    desc: 'Stage three (alignment): 1) humans rank multiple responses to the same prompt → train reward model r(x,y); 2) PPO updates the policy using the reward while a KL penalty anchors to the SFT distribution, preventing reward hacking. ChatGPT\'s direct source: turning a model from completer into a helpful, honest, harmless conversation partner.',
    latex: [
      '\\max_\\theta\\; \\mathbb{E}_{x\\sim D,\\,y\\sim\\pi_\\theta}\\big[r(x,y)\\big] - \\beta\\,\\mathrm{KL}\\big(\\pi_\\theta\\|\\pi_{\\text{SFT}}\\big)'
    ],
    shapes: ['preference pairs (x, y_w, y_l) → reward model','PPO: policy/value/reward/reference — four models'],
    code: "for (const {x, y} of ppoSamples) {\n  const adv = reward(x, y) - beta * kl(policy(x,y), ref(x,y));\n  update(policy, adv);   // PPO clipped objective\n}",
    params: 'reward model + policy + reference (~3-4x inference parameters)',
    refs: ['Ziegler et al. 2019 · Fine-Tuning Language Models from Human Preferences',
           'Ouyang et al. 2022 · Training language models to follow instructions (InstructGPT)',
           'Bai et al. 2022 · Training a Helpful and Harmless Assistant (Anthropic)']
  },
  'dpo': {
    title: 'DPO Direct Preference Optimization (2023)',
    desc: 'Folds RLHF\'s reward-model + PPO into one step: a closed-form transform from preference pairs proves the optimal reward can be written as a log-ratio between policy and reference model — alignment becomes a simple classification loss, no sampling, no explicit reward model, with much better stability and easier implementation. Variants: IPO, KTO, ORPO; industry runs RLHF/DPO side by side.',
    latex: [
      'L_{\\text{DPO}}=-\\log\\,\\sigma\\!\\Big(\\beta\\log\\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)}-\\beta\\log\\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\Big)'
    ],
    shapes: ['input is a preference pair (x, y_w, y_l)','one model + a frozen reference copy'],
    code: "const logRatio = (y) => policy(x, y) - ref(x, y);   // log π/π_ref\nconst loss = -sigmoid(beta * (logRatio(yw) - logRatio(yl)));",
    params: 'policy + frozen reference (2x inference parameters, no reward model)',
    refs: ['Rafailov et al. 2023 · Direct Preference Optimization',
           'Hong et al. 2024 · ORPO', 'Ethayarajh et al. 2024 · KTO']
  },
  'rlvr': {
    title: 'Verifiable-reward RL (R1-style reasoning training, 2024-25)',
    desc: 'The post-alignment new wave: in domains with objective answers (math, code), RL with verifiable rewards (answer checking, passing unit tests) makes models spontaneously develop long chains of thought, reflection and verification. DeepSeek-R1 proved pure RL (GRPO: group-relative advantages replace the critic) suffices for emergent reasoning; OpenAI\'s o1 series was the closed-source preview — test-time compute gained a training-side foundation.',
    latex: [
      'R=\\mathbb{1}[\\text{answer correct}],\\quad \\text{GRPO: } A_i=\\frac{r_i-\\text{mean}(r)}{\\text{std}(r)}\\ (\\text{group-relative})'
    ],
    shapes: ['prompt → long chain of thought → verifiable answer','rollouts can reach tens of thousands of tokens'],
    code: "for (const x of batch) {\n  const rs = sampleN(policy, x, G).map(y => verify(x, y)); // 0/1\n  const adv = (rs - mean(rs)) / std(rs);                    // GRPO\n  update(policy, adv);\n}",
    params: 'policy + reference (no critic)',
    refs: ['DeepSeek-AI 2025 · DeepSeek-R1: Incentivizing Reasoning via RL',
           'Shao et al. 2024 · DeepSeekMath (GRPO)',
           'OpenAI 2024 · Learning to Reason with LLMs (o1)']
  },
  'scaling': {
    title: 'Scaling Laws',
    desc: 'The fundamental law of the LLM era: validation loss falls as a smooth power law in parameters N, data D and compute C, extrapolable across orders of magnitude. Kaplan 2020 gave the first formulas; Chinchilla 2022 corrected that for fixed compute, parameters and data should scale proportionally (70B with 1.4T tokens) — rewriting industry training recipes and explaining both MoE (more params per FLOP) and the high-quality-data race.',
    latex: [
      'L(N)=L_\\infty+\\Big(\\frac{N_c}{N}\\Big)^{\\alpha},\\quad \\text{Chinchilla: } N_{\\text{opt}},D_{\\text{opt}}\\propto\\sqrt{C}'
    ],
    shapes: ['loss-vs-scale log-log straight line','predict large-model performance in advance'],
    code: "// fit power law on small-model series, extrapolate to target scale\nconst {alpha, Nc} = fitPowerLaw(sizes, losses);\nconst lossBig = Linf + Math.pow(Nc / N_big, alpha);",
    params: 'fitted parameters α, N_c (not model parameters)',
    refs: ['Kaplan et al. 2020 · Scaling Laws for Neural Language Models',
           'Hoffmann et al. 2022 · Training Compute-Optimal Large Language Models (Chinchilla)']
  },

  /* ============ 前传·基石（1943-2016） ============ */
  'mp': {
    title: 'MP Neuron (1943)',
    desc: 'McCulloch and Pitts abstracted the neuron into a logical unit of weighted sums + threshold firing and proved any Boolean function is realizable by composing such units — the first formalization of thought as computation. The ancestor of all neural networks.',
    latex: ['y=\\mathbb{1}\\Big[\\sum_i w_i x_i \\ge \\theta\\Big]'],
    shapes: ['inputs x_i ∈ {0,1} → binary output'],
    code: "const y = (w.reduce((a, wv, i) => a + wv * x[i], 0) >= theta) ? 1 : 0;",
    params: 'n weights + threshold θ (hand-set, not learnable)',
    refs: ['McCulloch & Pitts 1943 · A Logical Calculus of the Ideas Immanent in Nervous Activity']
  },
  'hebb': {
    title: 'Hebbian learning rule (1949)',
    desc: 'Neurons that fire together wire together: co-activated connections should strengthen — the first learning theory, purely local and teacher-free. Supervised, unsupervised and contrastive learning (SimCLR) are all echoes of this idea across eras.',
    latex: ['\\Delta w_i = \\eta\\, x_i\\, y'],
    shapes: ['connection strength w grows with co-activation'],
    code: "w[i] += eta * x[i] * y;   // fire together, wire together",
    params: 'learning rate η',
    refs: ['Hebb 1949 · The Organization of Behavior']
  },
  'perceptron': {
    title: 'Perceptron (1958)',
    desc: 'Rosenblatt gave the neuron learnable weights: an error-driven update rule + convergence theorem (guaranteed to stop on linearly separable data), and built dedicated hardware Mark I. The first machine that truly learned — its limits were exposed 11 years later by Perceptrons.',
    latex: ['w \\leftarrow w + \\eta\\,(t-y)\\,x'],
    shapes: ['linear decision boundary','linearly separable ⇒ guaranteed convergence'],
    code: "for (const {x, t} of samples)\n  w = w.add(x.mul(eta * (t - predict(x))));",
    params: 'n learnable weights + bias',
    refs: ['Rosenblatt 1958 · The Perceptron: A Probabilistic Model']
  },
  'winter': {
    title: '\'Perceptrons\' and the first AI winter (1969)',
    desc: 'Minsky and Papert mathematically proved a single-layer Perceptron cannot even represent XOR (not linearly separable); combined with scarce compute/data and the book\'s pessimism about multi-layer routes, funding dried up — the first AI winter. The cure (depth + nonlinearity + backprop) only arrived in 1986.',
    latex: ['XOR \\notin \\text{linearly separable} \\Rightarrow \\text{needs hidden layers + nonlinearity}'],
    shapes: ['single-layer decision boundary: one straight line','XOR needs two lines'],
    code: "// XOR needs a hidden layer:\nh = sigmoid(x.matmul(W1));\ny = sigmoid(h.matmul(W2));",
    params: '—',
    refs: ['Minsky & Papert 1969 · Perceptrons']
  },
  'neocognitron': {
    title: 'Neocognitron（1980）',
    desc: 'Fukushima\'s hierarchical network modeled the visual cortex: alternating feature-extraction and pooling layers progressively abstract edges → shapes → patterns, robust to translation — the direct intellectual ancestor of CNNs (LeNet/AlexNet).',
    latex: ['S layers (feature extraction) → C layers (pooling) → ⋯ alternating'],
    shapes: ['layer by layer: edges → parts → wholes'],
    code: "// hierarchical alternation (modern style)\nconst c1 = maxPool(conv2d(img, K1));\nconst c2 = maxPool(conv2d(c1, K2));",
    params: 'per-layer convolution kernels',
    refs: ['Fukushima 1980 · Neocognitron: A Self-organizing Neural Network Model']
  },
  'backprop': {
    title: 'Backpropagation (1986)',
    desc: 'Rumelhart, Hinton and Williams systematized the chain rule for multi-layer networks: errors propagate backward layer by layer, updating weights along the way — multi-layer networks finally became trainable, the first cornerstone of deep learning. Still the training core of every LLM today (with autodiff).',
    latex: [
      '\\frac{\\partial L}{\\partial w^{(l)}}=\\frac{\\partial L}{\\partial a^{(L)}}\\prod_{k>l}\\frac{\\partial a^{(k)}}{\\partial a^{(k-1)}}\\cdot\\frac{\\partial a^{(l)}}{\\partial w^{(l)}}'
    ],
    shapes: ['cache activations forward → backprop layer by layer'],
    code: "// cache activations forward, chain-rule backward\nlet delta = lossGrad;\nfor (let l = layers.length - 1; l >= 0; l--) {\n  grads[l] = delta.mul(activations[l].T);\n  delta = delta.matMul(layers[l].W.T).mul(actGrad[l]);\n}",
    params: '— (a training algorithm)',
    refs: ['Rumelhart, Hinton & Williams 1986 · Learning Representations by Back-propagating Errors']
  },
  'dbn': {
    title: 'DBN Deep Belief Network (2006)',
    desc: 'Hinton made deep networks trainable again via layer-wise unsupervised pre-training (stacked RBMs) + supervised fine-tuning — the start of the deep-learning revival. More profound still is the pre-train+fine-tune paradigm itself: GPT/BERT\'s training philosophy is its direct descendant.',
    latex: ['P(v|h)\\;\\text{layer-wise modeling} \\Rightarrow \\text{pre-train} \\to \\text{fine-tune}'],
    shapes: ['stacked RBMs layer by layer','pre-training → fine-tuning'],
    code: "// greedy layer-wise pre-training, then overall fine-tuning\nfor (const layer of layers) pretrain(layer, data);\nfineTune(layers, labels);",
    params: 'per-layer RBM parameters',
    refs: ['Hinton & Salakhutdinov 2006 · A Fast Learning Algorithm for Deep Belief Nets']
  },
  'alexnet': {
    title: 'AlexNet（2012）',
    desc: 'Krizhevsky, Sutskever and Hinton trained an 8-layer CNN on GPUs (ReLU + Dropout + augmentation) to sweep ImageNet (top-5 error 26%→15%): big data + GPU compute + deep networks aligned for the first time. The flashpoint of deep learning\'s golden age; the LLM compute roadmap started here.',
    latex: ['ReLU: \\max(0,x)  (less vanishing-gradient-prone than sigmoid)'],
    shapes: ['5 conv + 3 fc layers','60M parameters · 2× GTX 580'],
    code: "const y = tf.relu(conv2d(x, K1).maxPool()); // ReLU + pooling\nconst out = fc(y, Wfc);",
    params: '60 million',
    refs: ['Krizhevsky et al. 2012 · ImageNet Classification with Deep Convolutional Neural Networks']
  },
  'dqn': {
    title: 'DQN deep reinforcement learning（2013-15）',
    desc: 'DeepMind used CNN + experience replay + target networks to let an AI learn Atari games straight from screen pixels at human level — the beginning of deep RL. RL lineage: DQN (games) → AlphaGo → RLHF (alignment) → R1 (verifiable-reward reasoning); reinforcement learning worked its way to center stage in LLMs.',
    latex: [
      'Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma\\max_{a\'}Q(s\',a\')-Q(s,a)\\big]'
    ],
    shapes: ['pixels [84,84,4] → Q values [|A|]','experience replay buffer'],
    code: "const target = r + gamma * maxQ(nextState);\nupdate(Q, s, a, target);   // trained via experience-replay sampling",
    params: 'CNN ~10M params',
    refs: ['Mnih et al. 2013/2015 · Human-level Control through Deep Reinforcement Learning']
  },
  'adam': {
    title: 'Adam optimizer (2014)',
    desc: 'Bias-corrected first and second moment estimates give each parameter an adaptive learning rate: robust to hyperparameters, fast-converging, good by default. The de facto standard of deep learning — every LLM still uses its variant AdamW (decoupled weight decay).',
    latex: [
      'm_t=\\beta_1 m_{t-1}+(1-\\beta_1)g_t,\\;\\; v_t=\\beta_2 v_{t-1}+(1-\\beta_2)g_t^2',
      '\\theta\\leftarrow\\theta-\\eta\\,\\hat{m}_t/(\\sqrt{\\hat{v}_t}+\\varepsilon)'
    ],
    shapes: ['two moment states per parameter'],
    code: "m = b1*m + (1-b1)*g;          // first moment\nv = b2*v + (1-b2)*g.square(); // second moment\ntheta -= lr * m.hat().div(v.hat().sqrt().add(eps));",
    params: '2x state per parameter',
    refs: ['Kingma & Ba 2014 · Adam: A Method for Stochastic Optimization',
           'Loshchilov & Hutter 2017 · AdamW']
  },
  'bn': {
    title: 'Batch Normalization（2015）',
    desc: 'Per-mini-batch normalization + learnable scale/shift: faster training, more stable, larger learning rates usable. But it depends on batch statistics and breaks on sequence models/small batches — evolving into feature-wise LayerNorm (2016), then simplified into RMSNorm (2019): the historical starting point of Transformer normalization.',
    latex: [
      '\\hat{x}=\\frac{x-\\mu_B}{\\sqrt{\\sigma_B^2+\\varepsilon}}\\cdot\\gamma+\\beta'
    ],
    shapes: ['normalized along the batch dim [B,L,d]','running averages for inference'],
    code: "const mu = x.mean(0, true), var_ = x.variance(0, true);\nconst y = x.sub(mu).div(var_.add(eps).sqrt()).mul(g).add(b);",
    params: 'γ, β + statistics cache (2d)',
    refs: ['Ioffe & Szegedy 2015 · Batch Normalization',
           'Ba et al. 2016 · Layer Normalization (evolution)']
  },
  'gan': {
    title: 'GAN Generative Adversarial Network (2014)',
    desc: 'Generator G and discriminator D play a minimax game: G forges to fool D, D upgrades to catch it — through the duel G learns to make fakes indistinguishable. It opened the era of AI generative models. Diffusion shares the two great generative routes, and diffusion ultimately won on language (LLaDA/Mercury), but adversarial thinking lives on in many training tricks.',
    latex: [
      '\\min_G \\max_D\\; \\mathbb{E}[\\log D(x)]+\\mathbb{E}[\\log(1-D(G(z)))]'
    ],
    shapes: ['noise z → G → fakes → D → real/fake'],
    code: "const fake = G(z);\nconst dLoss = bce(D(x), 1) + bce(D(fake.detach()), 0);\nconst gLoss = bce(D(fake), 1);   // fool D",
    params: 'one set each for G and D',
    refs: ['Goodfellow et al. 2014 · Generative Adversarial Nets']
  },
  'resnet': {
    title: 'ResNet residual network (2016)',
    desc: 'Identity bypass y = F(x) + x: networks learn only residual corrections rather than full mappings, giving gradients an express lane — 152 layers stopped being a problem and degradation vanished. Transformer\'s Add & Norm inherited the idea the following year: every amber dashed bypass in these diagrams is this.',
    latex: [
      'y=\\mathcal{F}(x,\\{W_i\\})+x\\quad\\text{(identity mapping at zero cost)}'
    ],
    shapes: ['input and output same shape → add directly','152+ layers deep'],
    code: "const y = block(x).add(x);   // F(x) + x\n// in Transformers: h = ln(x.add(attn))",
    params: 'same as ordinary layers (zero parameters in the bypass)',
    refs: ['He et al. 2016 · Deep Residual Learning for Image Recognition']
  },
  'ebt': {
    title: 'Energy-Based Transformers（2025）',
    desc: 'Scores input-candidate pairs with energy function E(x,y); reasoning = iterative optimization descending the energy — building System-2 deliberation into the architecture itself rather than generating longer chains of thought. Same test-time-compute camp as TTT/Titans; claims of more stable training and better generalization await larger-scale validation — one of 2025\'s most watchable new directions.',
    latex: [
      'y^{(k+1)}=y^{(k)}-\\eta\\,\\nabla_y\\,E_\\theta(x,\\,y^{(k)}),\\quad y^{(0)}=\\text{initial guess}'
    ],
    shapes: ['each step: energy gradient descent','inference compute ↔ answer quality, adjustable'],
    code: "let y = initGuess(x);\nfor (let k = 0; k < K; k++)\n  y = y.sub(gradY(x, y).mul(eta));   // energy descent\nreturn y;",
    params: 'energy network E_θ',
    refs: ['Khona et al. 2025 · Energy-Based Transformers are Scalable Learners']
  },
  'v4': {
    title: 'DeepSeek V4: CSA sparse attention + mHC (Apr 2026)',
    desc: 'V4-Pro: 1.6T total / 49B active params (MIT open source), 1M context. Two architectural updates: 1) CSA compressed sparse attention + HCA compresses the 1M-context KV cache to ~10 percent of V3.2s; 2) mHC (manifold-constrained hyper-connections) widens residual pathways and adds manifold constraints, improving deep information flow. Three inference modes (Non-Think / Think High / Think Max) turn thinking depth into an API parameter.',
    latex: [
      '\\text{CSA：KV Cache}(1M)\\approx 10\\%\\times V3.2',
      'mHC:\\text{ widened residual pathways + manifold constraint}'
    ],
    shapes: ['1.6T total / 49B active params (MoE)','1M context'],
    code: "// switch thinking depth at the API layer\nconst out = v4(prompt, { think: 'high' });",
    params: '1.6T total / 49B active params',
    refs: ['DeepSeek-AI 2026 · DeepSeek-V4 Technical Report']
  },
  'k3': {
    title: 'Kimi K3: full-stack component replacement (Jul 2026)',
    desc: 'The first open-source 3T-class model (2.8T parameters, 896 experts choose 16, native vision, 1M context). The 2.8T productionization of the Kimi Linear paper with every component tuned for inference efficiency: KDA+gated MLA hybrid attention (69+24 layers), LatentMoE (pre-routing dimension reduction compresses communication), Attention Residuals (cross-layer residuals weighted by attention), full-stack NoPE — the first frontier model to drop RoPE entirely; positions carried implicitly by the causal mask and KDA state.',
    latex: [
      '\\text{K3}=69\\times\\text{KDA}+24\\times\\text{gated MLA}+\\text{LatentMoE}(896\\text{choose}16)+\\text{AttnRes}+\\text{NoPE}'
    ],
    shapes: ['2.8T total / ~32B active (1.8%)','native multimodal · 1M context'],
    code: "// 2026 mainstream: component-level swaps (inference-efficiency tuned)\nMoE       → LatentMoE;\nAttention → MLA + KDA linear hybrid;\nResidual  → Attention Residuals;\nRoPE      → NoPE;",
    params: '2.8T total / ~32B active params',
    refs: ['Moonshot AI 2026 · Kimi K3: Open Frontier Intelligence',
           'Raschka 2026 · Kimi K3 Architecture Notes']
  }
};

const ARCH_NODES = [
  /* ---- Encoder 列 ---- */
  { id:'input',    x:80,  y:48,  label:'Input Tokens',            sub:'token index sequence after tokenization', detail:'input',
    tip:'Token index sequence from source-sentence tokenization' },
  { id:'embed',    x:80,  y:112, label:'Embedding',              sub:'lookup then × √d_model', detail:'embed',
    tip:'Look up word vectors and scale so magnitudes match positional encoding' },
  { id:'pe',       x:80,  y:176, label:'+ Positional Encoding',  detail:'pe',
    tip:'Sinusoidal position encoding injects word order information' },
  { id:'enc-mha',  x:80,  y:262, label:'Multi-Head Self-Attention',           sub:'Self-Attention (fully parallel)', detail:'enc-mha',
    tip:'Every position attends to all positions in the sentence, done in one matrix op' },
  { id:'enc-an1',  x:80,  y:320, label:'Add & Norm',             detail:'addnorm',
    tip:'Residual connections stabilize gradients; layer norm stabilizes distributions' },
  { id:'enc-ffn',  x:80,  y:378, label:'Feed-Forward FFN',           sub:'d_model → d_ff → d_model', detail:'ffn',
    tip:'Per-position two-layer MLP; the main source of parameters' },
  { id:'enc-an2',  x:80,  y:436, label:'Add & Norm',             detail:'addnorm',
    tip:'Second residual + normalization sublayer' },
  /* ---- Decoder 列 ---- */
  { id:'dec-input',      x:458, y:48,  label:'Target input (shifted right)', sub:'starts with <bos>', detail:'input',
    tip:'Shifted right during training; generated prefix fed step-by-step at inference' },
  { id:'dec-embed',      x:458, y:112, label:'Embedding + position encoding', detail:'embed',
    tip:'Embedding and position encoding of the target sequence' },
  { id:'dec-masked-mha', x:458, y:198, label:'Masked Multi-Head Attention',     sub:'Only positions j ≤ i are visible', detail:'dec-masked-mha',
    tip:'Causal mask prevents peeking at future positions' },
  { id:'dec-an1',        x:458, y:256, label:'Add & Norm', detail:'addnorm', tip:'Residual + LayerNorm' },
  { id:'dec-cross',      x:458, y:314, label:'Cross-Attention', sub:'Q←Decoder, K/V←Encoder', detail:'dec-cross',
    tip:'The only gateway for the decoder to read source information' },
  { id:'dec-an2',        x:458, y:372, label:'Add & Norm', detail:'addnorm', tip:'Residual + LayerNorm' },
  { id:'dec-ffn',        x:458, y:430, label:'Feed-Forward FFN', detail:'ffn', tip:'Per-position two-layer MLP' },
  { id:'dec-an3',        x:458, y:488, label:'Add & Norm', detail:'addnorm', tip:'Residual + LayerNorm' },
  { id:'out',            x:458, y:568, label:'Linear + Softmax', sub:'project to vocab |V|', detail:'out',
    tip:'Last-position hidden state → full-vocab probability distribution' },
  { id:'outp',           x:458, y:632, label:'Output probability P(u|x)', sub:'argmax becomes the next input', detail:'out',
    tip:'Take the highest-probability token as next input, generating in a loop' }
];

const ARCH_FRAMES = [
  { x:62,  y:224, w:232, h:290, label:'Encoder Layer × N' },
  { x:440, y:160, w:232, h:372, label:'Decoder Layer × N' }
];

const ARCH_EDGES = [
  ['input','embed'], ['embed','pe'],
  ['enc-mha','enc-an1'], ['enc-an1','enc-ffn'], ['enc-ffn','enc-an2'],
  ['dec-input','dec-embed'],
  ['dec-masked-mha','dec-an1'], ['dec-an1','dec-cross'], ['dec-cross','dec-an2'],
  ['dec-ffn','dec-an3'],
  ['out','outp']
];

const ARCH_SPECIAL_EDGES = [
  /* pe → 编码器栈入口 */
  { d:'M 178 218 L 178 262', label:'' },
  /* 编码器输出 → 解码器交叉注意力（K/V 记忆） */
  { d:'M 276 457 C 350 457, 384 335, 458 335', label:'K, V (encoder memory)', lx:352, ly:378 },
  /* dec-embed → 解码器栈入口 */
  { d:'M 556 154 L 556 198', label:'' },
  /* dec-an3 → 输出层 */
  { d:'M 556 530 L 556 568', label:'' }
];

const ARCH_RESIDUALS = [
  'M 276 197 C 318 197, 318 341, 276 341',   /* pe → enc-an1 */
  'M 276 283 C 322 283, 322 457, 276 457',   /* enc-mha → enc-an2 */
  'M 654 219 C 700 219, 700 277, 654 277',   /* dec-masked-mha → dec-an1 */
  'M 654 335 C 706 335, 706 393, 654 393',   /* dec-cross → dec-an2 */
  'M 654 451 C 700 451, 700 509, 654 509'    /* dec-ffn → dec-an3 */
];

const ARCH_VIEW_PRE = {
  colTitles: [
    { x:330, y:92,  text:'① The recurrent era: state passes step by step; long-range dependency path O(t)' },
    { x:556, y:312, text:'② The attention revolution: any position connects directly; path O(1)' }
  ],
  nodes: [
    { id:'rnn', x:24, y:120, label:'Vanilla RNN (1990)', sub:'hₜ = φ(W·hₜ₋₁ + U·xₜ)', detail:'rnn',
      tip:'the origin of all sequence modeling: hidden state recurses through time' },
    { id:'lstm', x:282, y:120, label:'LSTM (1997) / GRU (2014)', sub:'gated memory mitigates vanishing gradients', detail:'lstm',
      tip:'Input/forget/output gates control cell state; GRU is its simplification' },
    { id:'s2s', x:540, y:120, label:'Seq2Seq (2014)', sub:'Encoder-Decoder + fixed-vector bottleneck', detail:'s2s',
      tip:'Established the MT paradigm, but squeezes the whole sentence into a fixed-length vector' },
    { id:'batt', x:282, y:340, label:'Additive attention (2015)', sub:'decoder progressively looks back at all encoder states', detail:'batt',
      tip:'Bahdanau attention: alignment weights αₜᵢ dynamically weight encoder states (Luong proposed the multiplicative version the same year)' },
    { id:'selfattn', x:540, y:340, label:'Self-attention (2017)', sub:'Q·K·V fully parallel → see 2017 Original', detail:'selfattn',
      tip:'Generalizes attention from decoder-looks-at-encoder to sequence-looks-at-itself, replacing recurrence' }
  ],
  frames: [],
  edges: [],
  special: [
    { d:'M 220 141 L 282 141' },                                  /* rnn → lstm */
    { d:'M 478 141 L 540 141' },                                  /* lstm → s2s */
    { d:'M 638 162 C 638 268, 380 268, 380 338' },                /* s2s → batt */
    { d:'M 478 361 L 540 361' }                                   /* batt → selfattn */
  ],
  residuals: []
};

const ARCH_VIEW_DENSE = {
  colTitles: [
    { x:380, y:26, text:'Decoder-Only · Dense（GPT → LLaMA → Qwen-Dense）' }
  ],
  nodes: [
    { id:'tok', x:282, y:36, label:'Input Tokens', sub:'tokenization → vocab indices', detail:'input',
      tip:'Token index sequence after BPE tokenization' },
    { id:'emb', x:282, y:92, label:'Embedding', sub:'lookup then × √d_model', detail:'embed',
      tip:'word-vector lookup then scaled (LLaMA onward drops the √d multiply)' },
    { id:'rope', x:282, y:148, label:'+ RoPE', sub:'rotary position encoding (absolute-position implementation, relative-position expression)', detail:'rope',
      tip:'Encodes position as 2D rotation; inner product depends only on relative position. ALiBi/YaRN are variants.' },
    { id:'rms1', x:282, y:218, label:'RMSNorm', sub:'Pre-Norm (normalize before sublayer)', detail:'rmsnorm',
      tip:'Pre-LN/RMSNorm stabilizes deep-network training, replacing Post-LN from 2017' },
    { id:'gqa', x:282, y:276, label:'GQA masked self-attention', sub:'causal mask · grouped-shared KV heads', detail:'gqa',
      tip:'Standard on Qwen/LLaMA-2+: fewer KV heads than Q heads shrinks the cache directly' },
    { id:'res1', x:282, y:334, label:'⊕ Residual', detail:'addnorm', tip:'direct residual; an express lane for gradients' },
    { id:'rms2', x:282, y:392, label:'RMSNorm', detail:'rmsnorm', tip:'second Pre-Norm' },
    { id:'swiglu', x:282, y:450, label:'SwiGLU FFN', sub:'gating: d → d_ff → d', detail:'swiglu',
      tip:'SiLU gated branch ⊙ linear branch, replacing ReLU FFN' },
    { id:'res2', x:282, y:508, label:'⊕ Residual', detail:'addnorm', tip:'second residual' },
    { id:'lnf', x:282, y:580, label:'Final RMSNorm', detail:'rmsnorm', tip:'final normalization before output' },
    { id:'head', x:282, y:638, label:'LM Head · Softmax(|V|)', sub:'often weight-tied with the embedding', detail:'lm-head',
      tip:'hidden state → full-vocab distribution; autoregressive token by token' }
  ],
  frames: [
    { x:264, y:196, w:232, h:376, label:'Decoder Block × N（Pre-Norm）' }
  ],
  edges: [
    ['tok','emb'], ['emb','rope'], ['rope','rms1'], ['rms1','gqa'], ['gqa','res1'],
    ['res1','rms2'], ['rms2','swiglu'], ['swiglu','res2'], ['res2','lnf'], ['lnf','head']
  ],
  special: [],
  residuals: [
    'M 478 190 C 540 190, 540 355, 478 355',   /* rope → res1 */
    'M 478 355 C 540 355, 540 529, 478 529'    /* res1 → res2 */
  ]
};

const ARCH_VIEW_MOE = {
  colTitles: [
    { x:330, y:26, text:'Decoder-Only · MoE（Switch → Mixtral → DeepSeek-V3 → Qwen3 → K2）' }
  ],
  nodes: [
    { id:'tok', x:232, y:36, label:'Input Tokens', sub:'tokenization → vocab indices', detail:'input', tip:'same as the Dense view' },
    { id:'emb', x:232, y:92, label:'Embedding', detail:'embed', tip:'word-vector lookup' },
    { id:'rope', x:232, y:148, label:'+ RoPE', detail:'rope', tip:'rotary position encoding' },
    { id:'rms1', x:232, y:218, label:'RMSNorm', sub:'Pre-Norm', detail:'rmsnorm', tip:'Pre-Norm normalization' },
    { id:'gqa', x:232, y:276, label:'GQA masked self-attention', detail:'gqa', tip:'Causal self-attention with grouped-shared KV' },
    { id:'an1', x:232, y:334, label:'Add & Norm', detail:'addnorm', tip:'attention residual + normalization' },
    { id:'rms2', x:232, y:392, label:'RMSNorm', detail:'rmsnorm', tip:'Pre-Norm of the MoE sublayer' },
    { id:'router', x:232, y:450, label:'MoE router gating', sub:'softmax → top-K sparse activation', detail:'moe-router',
      tip:'each token picks K experts only: decoupling parameter scale from compute' },
    { id:'wsum', x:232, y:508, label:'Sparse weighted synthesis', sub:'y = Σ gᵢ · Expertᵢ(x)', detail:'moe-sum',
      tip:'only selected experts\' outputs join the synthesis' },
    { id:'an2', x:232, y:566, label:'Add & Norm', detail:'addnorm', tip:'MoE residual + normalization' },
    { id:'lnf', x:232, y:624, label:'Final RMSNorm + LM Head', sub:'Linear → Softmax(|V|)', detail:'lm-head',
      tip:'output layer (DeepSeek-V3\'s MTP heads omitted)' },
    { id:'exp1', x:532, y:406, label:'Expert 1 · SwiGLU', detail:'moe-expert', tip:'one expert = one SwiGLU FFN' },
    { id:'exp2', x:532, y:454, label:'Expert 2 · SwiGLU', detail:'moe-expert', tip:'one expert = one SwiGLU FFN' },
    { id:'exp3', x:532, y:502, label:'Expert 3 · SwiGLU', detail:'moe-expert', tip:'one expert = one SwiGLU FFN' },
    { id:'exp4', x:532, y:550, label:'Expert E · SwiGLU', sub:'(4 shown for illustration; actually 64~384)', detail:'moe-expert',
      tip:'Qwen3: 128 choose 8 · K2: 384 choose 8 · DeepSeek-V3: 256 choose 8 + 1 shared' }
  ],
  frames: [
    { x:214, y:196, w:232, h:394, label:'Decoder Block × N' },
    { x:514, y:388, w:232, h:204, label:'Experts × E · Top-K activation' }
  ],
  edges: [
    ['tok','emb'], ['emb','rope'], ['rope','rms1'], ['rms1','gqa'], ['gqa','an1'],
    ['an1','rms2'], ['rms2','router'], ['router','wsum'], ['wsum','an2'], ['an2','lnf']
  ],
  special: [
    { d:'M 428 462 C 486 462, 486 427, 532 427' },
    { d:'M 428 468 C 486 468, 486 475, 532 475' },
    { d:'M 428 474 C 486 474, 486 523, 532 523' },
    { d:'M 428 480 C 486 480, 486 571, 532 571' },
    { d:'M 532 427 C 474 427, 474 520, 428 520' },
    { d:'M 532 475 C 474 475, 474 526, 428 526' },
    { d:'M 532 523 C 474 523, 474 532, 428 532' },
    { d:'M 532 571 C 474 571, 474 538, 428 538' }
  ],
  residuals: [
    'M 478 190 C 520 190, 520 355, 478 355',   /* rope → an1 */
    'M 478 355 C 520 355, 520 587, 478 587'    /* an1 → an2 */
  ]
};

const ARCH_VIEW_ATTN = {
  nodeW: 160,
  colTitles: [
    { x:370, y:90,  text:'Route A · KV cache compression (inference memory/bandwidth)' },
    { x:370, y:300, text:'Route B · compute and memory-access efficiency (long context)' },
    { x:370, y:510, text:'Route C · context extension and training stabilization' }
  ],
  nodes: [
    { id:'mha-ev', x:20, y:120, label:'MHA multi-head attention', sub:'2017 · h independent KV groups', detail:'mha-ev',
      tip:'baseline: KV cache = 2·L·h·dₕ' },
    { id:'mqa', x:200, y:120, label:'MQA multi-query', sub:'2019 · all heads share 1 KV group', detail:'mqa',
      tip:'Shazeer：One Write-Head is All You Need' },
    { id:'gqa', x:380, y:120, label:'GQA grouped-query', sub:'2023 · grouped sharing (Qwen family)', detail:'gqa',
      tip:'MQA-MHA compromise via uptraining; standard on LLaMA-2/3 and Qwen' },
    { id:'mla', x:560, y:120, label:'MLA latent attention', sub:'2024 · low-rank compression (DeepSeek)', detail:'mla',
      tip:'Cache stores only latent vector c; K/V restored on demand via up-projection; adopted by K2' },
    { id:'flash', x:20, y:330, label:'FlashAttention', sub:'2022 · tiled online softmax', detail:'flash',
      tip:'IO-aware kernel: math unchanged, memory O(L²)→O(L); v2/v3 keep evolving' },
    { id:'sparse', x:200, y:330, label:'Sparse attention', sub:'2025 · NSA / MoBA', detail:'sparse',
      tip:'compute only important blocks with trainable sparsity: DeepSeek NSA, Kimi MoBA' },
    { id:'linear', x:380, y:330, label:'Linear attention', sub:'2020-25 · kernelized O(Ld²)', detail:'linear',
      tip:'φ(Q)(φ(K)ᵀV): linear in sequence length — see the Linear·SSM view' },
    { id:'hybrid', x:560, y:330, label:'Hybrid architecture', sub:'Jamba · Griffin · KDA', detail:'hybrid',
      tip:'A few full-attention layers mixed proportionally with many linear layers (Kimi Linear etc.)' },
    { id:'swa', x:20, y:540, label:'Sliding-window SWA', sub:'2023 · 5:1 local/global interleaving', detail:'swa',
      tip:'Mistral/Gemma: each layer sees window w only, with sparse global layers' },
    { id:'ctxext', x:200, y:540, label:'Context extension', sub:'PI · NTK · YaRN', detail:'ctx-ext',
      tip:'RoPE training length → million-token inference: frequency-wise interpolation/extrapolation' },
    { id:'qknorm', x:380, y:540, label:'Training stabilization', sub:'QK-Norm · soft-cap · z-loss', detail:'qknorm',
      tip:'The three great stabilizers of LLM training (Gemma 2/3, Chameleon)' },
    { id:'diff', x:560, y:540, label:'Differential attention', sub:'2024 · dual-softmax denoising', detail:'diff',
      tip:'DIFF Transformer: differential-amplifier idea, hallucination-resistant' }
  ],
  frames: [],
  edges: [],
  special: [
    { d:'M 180 141 L 200 141' },
    { d:'M 360 141 L 380 141' },
    { d:'M 540 141 L 560 141' },
    { d:'M 100 162 C 100 290, 60 290, 62 328' },   /* mha-ev ↓ flash */
    { d:'M 180 351 L 200 351' },                   /* flash → sparse */
    { d:'M 360 351 L 380 351' },                   /* sparse → linear */
    { d:'M 540 351 L 560 351' },                   /* linear → hybrid */
    { d:'M 100 372 L 100 538' },                   /* flash ↓ swa */
    { d:'M 280 372 L 280 538' }                    /* sparse ↓ ctxext */
  ],
  residuals: []
};

const ARCH_VIEW_SSM = {
  colTitles: [
    { x:178, y:60, text:'Branch 1 · kernelized linear attention' },
    { x:556, y:60, text:'Branch 2 · structured state-space models (SSM)' }
  ],
  nodes: [
    { id:'lin', x:80, y:100, label:'Linear Transformer (2020)', sub:'Transformers are RNNs', detail:'linear',
      tip:'φ kernelization + associativity escape L²' },
    { id:'rwkv', x:80, y:210, label:'RWKV (2022)', sub:'channel-wise decay · parallelizable and recurrent', detail:'rwkv',
      tip:'Rewrites attention as a decayed RNN' },
    { id:'xlstm', x:80, y:320, label:'xLSTM (2024)', sub:'exponential memory + matrix state', detail:'xlstm',
      tip:'The LSTM originators revival work' },
    { id:'s4', x:458, y:100, label:'S4 (2021)', sub:'HiPPO + structured SSM', detail:'s4',
      tip:'continuous state space + kernelized convolution' },
    { id:'mamba', x:458, y:210, label:'Mamba (2023)', sub:'selective mechanism · hardware-aware scan', detail:'mamba',
      tip:'Makes SSM parameters input-dependent → content-based filtering; Mamba-2 (2024) unifies with attention' },
    { id:'kda', x:269, y:440, label:'Kimi Linear · KDA (2025)', sub:'gated incremental memory for 3/4 of layers + MLA for 1/4', detail:'kda',
      tip:'Kimis hybrid linear architecture: million-token context with wins on both memory and throughput' }
  ],
  frames: [],
  edges: [
    ['lin','rwkv'], ['rwkv','xlstm'], ['s4','mamba']
  ],
  special: [
    { d:'M 178 362 C 178 410, 300 410, 330 438' },   /* xlstm → kda */
    { d:'M 556 252 C 556 380, 434 380, 404 438' }    /* mamba → kda */
  ],
  residuals: []
};

const ARCH_VIEW_GEN = {
  colTitles: [
    { x:380, y:26, text:'Generation paradigms: autoregression and its challengers' }
  ],
  nodes: [
    { id:'autoreg', x:282, y:48, label:'Autoregressive NTP (mainstream)', sub:'GPT family · left-to-right token by token', detail:'autoreg',
      tip:'P(y)=∏P(yₜ|y<ₜ): the training and generation paradigm of all mainstream LLMs' },
    { id:'mtp', x:60, y:210, label:'MTP multi-token prediction', sub:'DeepSeek-V3 · predicts multiple steps at once', detail:'mtp',
      tip:'Auxiliary heads densify training signal; usable as speculative drafts at inference' },
    { id:'spec', x:504, y:210, label:'Speculative decoding', sub:'draft model + parallel verification (EAGLE)', detail:'spec',
      tip:'Math-lossless 2-4x decoding speedup; belongs to the inference-system layer, not architecture',
      dashed:true, badge:'Inference layer' },
    { id:'diffusion', x:282, y:380, label:'Diffusion language models', sub:'LLaDA · Mercury parallel denoising', detail:'diffusion',
      tip:'Iterative denoising from full mask: breaks left-to-right dependence' }
  ],
  frames: [],
  edges: [],
  special: [
    { d:'M 300 90 C 220 130, 180 170, 158 208' },              /* autoreg → mtp */
    { d:'M 460 90 C 540 130, 580 170, 588 208' },              /* autoreg → spec */
    { d:'M 256 231 L 504 231', label:'Draft / verify', lx:380, ly:224 },
    { d:'M 380 90 L 380 378',  label:'Paradigm challenger · parallel denoising', lx:462, ly:300 }
  ],
  residuals: []
};

const ARCH_VIEW_FRONTIER = {
  colTitles: [
    { x:178, y:60, text:'Detokenization and memory layers' },
    { x:556, y:60, text:'Test-time compute and depth' },
    { x:380, y:420, text:'Modality extension: everything is a token' }
  ],
  nodes: [
    { id:'blt', x:80, y:100, label:'BLT byte-level LM', sub:'2024 · entropy-driven dynamic patches', detail:'blt',
      tip:'Meta: drop BPE, allocate compute by information entropy' },
    { id:'memlayer', x:80, y:210, label:'Memory layers', sub:'2024 · sparse key-value replaces FFN', detail:'memlayer',
      tip:'Store knowledge in memory, reason via attention; large factuality gains' },
    { id:'looped', x:458, y:100, label:'Deep looping / latent reasoning', sub:'2018→25 · adaptive recurrent depth', detail:'looped',
      tip:'Universal Transformer returns: loop more on hard problems' },
    { id:'ttt', x:458, y:210, label:'TTT layer', sub:'2024 · hidden state = online learner', detail:'ttt',
      tip:'Sequence modeling is test-time training' },
    { id:'titans', x:269, y:330, label:'Titans neural memory', sub:'2025 · surprise-driven · tens-of-millions context', detail:'titans',
      tip:'Google: long-term memory and short-term attention divide the labor' },
    { id:'multimodal', x:269, y:440, label:'Multimodal fusion', sub:'LLaVA · Chameleon · native Omni', detail:'multimodal',
      tip:'Vision/audio unified into tokens for the Transformer' },
    { id:'ebt', x:470, y:330, label:'EBT Energy-based Transformer', sub:'2025 · implicit reasoning via energy functions', detail:'ebt',
      tip:'Reasoning = iterative descent on energy minimization; a new test-time-compute branch' }
  ],
  frames: [],
  edges: [
    ['ttt','titans']
  ],
  special: [
    { d:'M 178 252 C 178 300, 300 300, 340 328' }   /* memlayer → titans */
  ],
  residuals: []
};

const ARCH_VIEW_TRAIN = {
  colTitles: [
    { x:330, y:60,  text:'The three-stage paradigm: pre-training → instruction tuning → alignment' },
    { x:330, y:200, text:'key branches and underlying laws' }
  ],
  nodes: [
    { id:'pretrain', x:24, y:100, label:'Pre-training',
      sub:'self-supervised NTP/MLM · trillion tokens', detail:'pretrain',
      tip:'source of language/knowledge/reasoning foundations; alignment adds no new capabilities' },
    { id:'sft', x:282, y:100, label:'SFT instruction tuning',
      sub:'2021/22 · supervised learning from human demonstrations', detail:'sft',
      tip:'teaches the base model to follow instructions; data quality >> quantity (LIMA)' },
    { id:'rlhf', x:540, y:100, label:'RLHF reinforcement learning',
      sub:'2022 · reward model + PPO alignment', detail:'rlhf',
      tip:'from completion to conversation — the direct ancestor of ChatGPT' },
    { id:'scaling', x:24, y:240, label:'Scaling Laws',
      sub:'2020/22 · Kaplan / Chinchilla', detail:'scaling',
      tip:'loss falls with scale as a power law; parameters and data scale proportionally' },
    { id:'rlvr', x:282, y:240, label:'Verifiable-reward RL',
      sub:'2024-25 · R1-style long chain-of-thought training', detail:'rlvr',
      tip:'math/code reward from standard answers; GRPO needs no critic' },
    { id:'dpo', x:540, y:240, label:'DPO direct preference optimization',
      sub:'2023 · no explicit reward model', detail:'dpo',
      tip:'folds RLHF\'s two steps into one classification-style loss' }
  ],
  frames: [],
  edges: [],
  special: [
    { d:'M 220 121 L 282 121' },   /* pretrain → sft */
    { d:'M 478 121 L 540 121' },   /* sft → rlhf */
    { d:'M 122 162 L 122 238' },   /* pretrain ↓ scaling */
    { d:'M 380 162 L 380 238' },   /* sft ↓ rlvr */
    { d:'M 638 162 L 638 238' }    /* rlhf ↓ dpo */
  ],
  residuals: []
};

const ARCH_VIEW_DAWN = {
  nodeW: 160,
  colTitles: [
    { x:370, y:60,  text:'Theoretical foundations 1943-1969' },
    { x:370, y:230, text:'Revival and breakthroughs 1980-2012' },
    { x:370, y:400, text:'Modern cornerstones 2014-2016' }
  ],
  nodes: [
    { id:'mp', x:24, y:110, label:'MP Neuron (1943)',
      sub:'weighted sum + threshold firing', detail:'mp',
      tip:'The mathematical origin of neural networks: any Boolean function can be composed from them' },
    { id:'hebb', x:208, y:110, label:'Hebb rule (1949)',
      sub:'co-activation → stronger connections', detail:'hebb',
      tip:'The first learning theory; the intellectual source of unsupervised/contrastive learning' },
    { id:'perceptron', x:392, y:110, label:'Perceptron (1958)',
      sub:'a learnable linear classifier', detail:'perceptron',
      tip:'The first machine that truly learned (it was hardware at the time)' },
    { id:'winter', x:576, y:110, label:'《Perceptrons》(1969)',
      sub:'XOR not separable → AI winter', detail:'winter',
      tip:'One layer is not enough: needs depth + nonlinearity (added in 1986)' },
    { id:'neocognitron', x:24, y:280, label:'Neocognitron (1980)',
      sub:'hierarchical feature extraction → precursor of CNNs', detail:'neocognitron',
      tip:'Mimics visual-cortex alternation: convolution + pooling' },
    { id:'backprop', x:208, y:280, label:'Backpropagation (1986)',
      sub:'chain rule backpropagates layer by layer', detail:'backprop',
      tip:'Deep networks finally became trainable — still the training core of every LLM today' },
    { id:'dbn', x:392, y:280, label:'DBN (2006)',
      sub:'layer-wise pre-training + fine-tuning', detail:'dbn',
      tip:'Where the deep-learning revival began; the first victory of the pre-train+fine-tune paradigm' },
    { id:'alexnet', x:576, y:280, label:'AlexNet (2012)',
      sub:'ignited by GPUs + big data', detail:'alexnet',
      tip:'The three pillars aligned for the first time; the LLM compute roadmap started here' },
    { id:'adam', x:24, y:450, label:'Adam (2014)',
      sub:'adaptive moment optimizer', detail:'adam',
      tip:'Still the default optimizer of all LLMs (AdamW variants)' },
    { id:'gan', x:208, y:450, label:'GAN (2014)',
      sub:'adversarial generation · where generative models began', detail:'gan',
      tip:'Adversarial/generative thinking → diffusion models (LLaDA/Mercury)' },
    { id:'bn', x:392, y:450, label:'BatchNorm (2015)',
      sub:'→ LayerNorm → RMSNorm', detail:'bn',
      tip:'Start of the normalization evolution line; sequence models ultimately chose RMSNorm' },
    { id:'resnet', x:576, y:450, label:'ResNet (2016)',
      sub:'residual bypass → Transformer Add', detail:'resnet',
      tip:'Every amber dashed bypass in the diagrams is this idea' }
  ],
  frames: [],
  edges: [],
  special: [
    { d:'M 184 131 L 208 131' }, { d:'M 368 131 L 392 131' },
    { d:'M 552 131 L 576 131' },
    { d:'M 184 301 L 208 301' }, { d:'M 368 301 L 392 301' },
    { d:'M 552 301 L 576 301' },
    { d:'M 184 471 L 208 471' }, { d:'M 368 471 L 392 471' },
    { d:'M 552 471 L 576 471' }
  ],
  residuals: []
};

const ARCH_VIEWS = {
  pre:   ARCH_VIEW_PRE,
  orig:  { colTitles:[{x:178,y:26,text:'Encoder'},{x:556,y:26,text:'Decoder'}],
           frames:ARCH_FRAMES, nodes:ARCH_NODES, edges:ARCH_EDGES,
           special:ARCH_SPECIAL_EDGES, residuals:ARCH_RESIDUALS },
  dense: ARCH_VIEW_DENSE,
  moe:   ARCH_VIEW_MOE,
  attn:  ARCH_VIEW_ATTN,
  ssm:   ARCH_VIEW_SSM,
  gen:   ARCH_VIEW_GEN,
  frontier: ARCH_VIEW_FRONTIER,
  train: ARCH_VIEW_TRAIN,
  dawn:  ARCH_VIEW_DAWN
};

const ERAS = [
  { from:1943, to:1989, label:'Phase I · Birth of neural networks (1943-1986)',
    meaning:'Computers gain neurons; a way to train multi-layer networks is found' },
  { from:1990, to:2016, label:'Phase II · Deep learning revival and breakthroughs (1990-2016)',
    meaning:'Data, algorithms and compute align; deep learning enters its golden age' },
  { from:2017, to:2022, label:'Phase III · The Transformer era (2017-2022)',
    meaning:'Model scale explodes; Transformer becomes the universal backbone' },
  { from:2023, to:2100, label:'Phase IV · Post-Transformer exploration (2023-2026)',
    meaning:'Pursuing efficiency, long context and low cost on the road to general intelligence' }
];

const HISTORY = [
  /* ---- 第一阶段：神经网络诞生 1943-1986 ---- */
  { year:1943, view:'dawn', name:'MP Neuron',
    paper:'A Logical Calculus of the Ideas Immanent in Nervous Activity (McCulloch & Pitts)',
    contrib:'logical calculus of nervous activity; theoretical foundation of artificial neural networks',
    highlights:['mp'],
    note:'Where it all began: formalizing thought as weighted sums + threshold firing.' },
  { year:1949, view:'dawn', name:'Hebbian learning rule',
    paper:'The Organization of Behavior (Hebb)',
    contrib:'co-activation strengthens neuron connections: the first learning rule',
    highlights:['hebb','mp'],
    note:'Fire together, wire together — the intellectual source of unsupervised and contrastive learning.' },
  { year:1958, view:'dawn', name:'Perceptron',
    paper:'The Perceptron (Rosenblatt)',
    contrib:'learnable weights + convergence theorem; opened the field of neural network research',
    highlights:['perceptron'],
    note:'The first machine that truly learned (hardware Mark I); media once predicted machines would walk, talk and be conscious.' },
  { year:1969, view:'dawn', name:'\'Perceptrons\' and the winter',
    paper:'Perceptrons (Minsky & Papert)',
    contrib:'proved single layers cannot represent XOR, prompting theoretical reflection',
    highlights:['winter'],
    note:'One book froze a field: funding dried up, the first AI winter — the cure (depth + backprop) took 17 years to arrive.' },
  { year:1980, view:'dawn', name:'Neocognitron',
    paper:'Neocognitron: A Self-organizing Neural Network Model (Fukushima)',
    contrib:'hierarchical neural network; foundation of convolutional thinking',
    highlights:['neocognitron'],
    note:'Alternating feature-extraction and pooling layers — LeNet repeated this a decade later, AlexNet two decades later.' },
  { year:1986, view:'dawn', name:'Backpropagation',
    paper:'Learning Representations by Back-propagating Errors (Rumelhart, Hinton & Williams)',
    contrib:'chain rule propagates backward layer by layer; multi-layer networks train effectively',
    highlights:['backprop'],
    note:'The winter\'s cure finally arrived: multi-layer networks became trainable for the first time. Still the core of every LLM\'s training today.' },
  /* ---- 前史：循环时代 ---- */
  { year:1990, view:'pre', name:'Vanilla RNN',
    paper:'Finding Structure in Time (Elman)',
    contrib:'the origin of all sequence modeling: hidden state recurses through time',
    highlights:['rnn'],
    note:'The most primitive architecture: step t must wait for t−1, and gradient products limit memory to ~10-20 steps — parallelism and long-range dependencies were the twin obstacles for 30 years.' },
  { year:1997, view:'pre', name:'LSTM',
    paper:'Long Short-Term Memory (Hochreiter & Schmidhuber)',
    contrib:'gated memory cell eases vanishing gradients in recurrent networks',
    highlights:['rnn','lstm'],
    note:'A conveyor belt for the RNN: cell state flows near-linearly. It ruled sequence modeling for 15 years (GRU is its 2014 simplification).' },
  { year:1998, view:'dawn', name:'LeNet (CNN in production)',
    paper:'Gradient-Based Learning Applied to Document Recognition (LeCun et al.)',
    contrib:'CNN lands on handwritten digit recognition',
    highlights:['neocognitron'],
    note:'CNN\'s first industrial application (US check recognition) — an early demonstration of end-to-end learning replacing feature engineering.' },
  { year:2006, view:'dawn', name:'DBN Deep Belief Network',
    paper:'A Fast Learning Algorithm for Deep Belief Nets (Hinton & Salakhutdinov)',
    contrib:'layer-wise pre-training: an efficient training algorithm for deep networks',
    highlights:['dbn','backprop'],
    note:'The start of the deep-learning revival; the first victory of the pre-train + fine-tune paradigm — GPT/BERT are its spiritual descendants.' },
  { year:2012, view:'dawn', name:'AlexNet',
    paper:'ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al.)',
    contrib:'GPU+ReLU+Dropout won ImageNet and ignited deep learning',
    highlights:['alexnet','backprop'],
    note:'Data, algorithms and compute aligned for the first time — the foundation of every large-model story since.' },
  { year:2013, view:'train', name:'DQN deep reinforcement learning',
    paper:'Playing Atari with Deep Reinforcement Learning (Mnih et al.)',
    contrib:'learning games from raw pixels: deep RL begins',
    highlights:['rlvr'],
    note:'RL lineage: DQN → AlphaGo → RLHF → R1 — reinforcement learning worked its way to center stage in LLMs.' },
  { year:2014, view:'pre', name:'Seq2Seq + GRU',
    paper:'Sequence to Sequence Learning (Sutskever et al.) · Learning Phrase Representations (Cho et al.)',
    contrib:'the encoder-decoder framework established; machine translation goes neural',
    highlights:['lstm','s2s'],
    note:'A general variable-length-to-variable-length paradigm was born; but the whole sentence is squeezed into one fixed vector and long sentences overflow — bottleneck as opportunity.' },
  { year:2014, view:'dawn', name:'GAN generative adversarial nets',
    paper:'Generative Adversarial Nets (Goodfellow et al.)',
    contrib:'opened the era of AI generative models',
    highlights:['gan'],
    note:'Game-theoretic generator-vs-discriminator training — the intellectual ancestor of the diffusion-LM generation line.' },
  { year:2014, view:'dawn', name:'Adam optimizer',
    paper:'Adam: A Method for Stochastic Optimization (Kingma & Ba)',
    contrib:'adaptive moment optimization greatly improved training efficiency',
    highlights:['adam'],
    note:'Good-by-default optimizer: from Perceptron to GPT it has been deep learning\'s de facto standard (LLMs use AdamW variants).' },
  { year:2015, view:'pre', name:'Attention mechanism',
    paper:'Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al.)',
    contrib:'decoder progressively looks back at all encoder states; soft alignment',
    highlights:['s2s','batt','selfattn'],
    note:'The origin of attention: α weights dynamically weight encoder states (Luong proposed the dot-product version the same year). One step short of generalizing attention to the sequence itself.' },
  /* ---- Transformer 与三大范式 ---- */
  { year:2015, view:'attn', name:'BatchNorm',
    paper:'Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy)',
    contrib:'normalization accelerates training and improves stability',
    highlights:['qknorm'],
    note:'Start of the normalization line: BN (batch-wise) → LayerNorm (feature-wise) → RMSNorm — Transformers ultimately chose RMSNorm mainly because BN is unfriendly to sequences/small batches.' },
  { year:2016, view:'dense', name:'ResNet residual network',
    paper:'Deep Residual Learning for Image Recognition (He et al.)',
    contrib:'residual connections solve deep-network degradation',
    highlights:['res1','res2'],
    note:'y = F(x) + x: an express lane for gradients. Transformer\'s Add & Norm inherited it the following year — every amber dashed line you see in the Dense view is this.' },
  { year:2017, view:'orig', name:'Transformer', primary:true,
    paper:'Attention Is All You Need (Vaswani et al.)',
    contrib:'attention only, no recurrence or convolution; fully parallel training',
    highlights:['embed','pe','enc-mha','enc-an1','enc-ffn','enc-an2','dec-masked-mha','dec-cross'],
    note:'The watershed: self-attention connects any two positions at O(1). It then split into three paradigms — Encoder-only (BERT), Decoder-only (GPT), full Enc-Dec (T5).' },
  { year:2018, view:'orig', name:'GPT-1 (the Decoder-Only beginning)',
    paper:'Improving Language Understanding by Generative Pre-Training (Radford et al.)',
    contrib:'Decoder-only + generative pre-training + fine-tuning',
    highlights:['dec-input','dec-embed','dec-masked-mha','dec-an1','out','outp'],
    note:'Decoder-Only can do understanding tasks too: 12 blocks + LM head, autoregressive objective all the way — the start of the GPT roadmap.' },
  { year:2018, view:'orig', name:'BERT（Encoder-Only）',
    paper:'BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.)',
    contrib:'bidirectional encoder pre-train + fine-tune paradigm swept NLU leaderboards',
    highlights:['embed','pe','enc-mha','enc-an1','enc-ffn','enc-an2'],
    note:'Encoder-only: bidirectional context suits understanding tasks (RoBERTa 2019 refined the recipe further).' },
  { year:2019, view:'orig', name:'GPT-2',
    paper:'Language Models are Unsupervised Multitask Learners (Radford et al.)',
    contrib:'1.5B-parameter decoder LM; zero-shot multitask; Pre-LN established',
    highlights:['dec-input','dec-embed','dec-masked-mha','dec-an1','dec-ffn','dec-an3','out','outp'],
    note:'Decoder-only: causal-masked autoregressive generation; scale is capability. Pre-LN replaced Post-LN and deep training became stable.' },
  { year:2019, view:'orig', name:'T5 (Enc-Dec revival)',
    paper:'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al.)',
    contrib:'everything is text-to-text; scaled validation of Encoder-Decoder',
    highlights:['input','embed','enc-mha','dec-cross','out'],
    note:'The culmination of the Encoder-Decoder line (11B); translation/summarization and other input-output tasks still commonly use this paradigm.' },
  { year:2019, view:'train', name:'Origins of RLHF',
    paper:'Fine-Tuning Language Models from Human Preferences (Ziegler et al.)',
    contrib:'first systematization: human preferences → reward → reinforcement learning',
    highlights:['rlhf'],
    note:'The common origin of OpenAI and Anthropic: aligning models to human preferences, three years before scaling bigger became the focus.' },
  { year:2019, view:'attn', name:'MQA multi-query attention',
    paper:'Fast Transformer Decoding: One Write-Head is All You Need (Shazeer)',
    contrib:'all heads share one KV group; inference cache shrinks h-fold',
    highlights:['mha-ev','mqa'],
    note:'The first shot of the efficiency-evolution line: finding the inference bottleneck in the KV cache, not parameters — MQA→GQA→MLA followed in one lineage.' },
  { year:2020, view:'orig', name:'GPT-3',
    paper:'Language Models are Few-Shot Learners (Brown et al.)',
    contrib:'175B parameters; in-context few-shot learning',
    highlights:['dec-input','dec-masked-mha','out','outp'],
    note:'Same architecture as GPT-2; scale made in-context learning emerge — the start of the large-model era (ChatGPT is its RLHF descendant).' },
  { year:2020, view:'train', name:'Scaling Laws',
    paper:'Scaling Laws for Neural Language Models (Kaplan et al.)',
    contrib:'loss falls as a power law in params/data/compute, extrapolable across orders of magnitude',
    highlights:['scaling'],
    note:'The quantitative basis for brute force at scale: large-model performance is predictable before training — Chinchilla (2022) corrected it to proportional scaling of parameters and data.' },
  { year:2020, view:'orig', name:'ViT (cross-domain)',
    paper:'An Image is Worth 16x16 Words (Dosovitskiy et al.)',
    contrib:'split images into patches as tokens: Transformer unifies vision',
    highlights:['input','embed','enc-mha','enc-an1'],
    note:'Proved self-attention is modality-agnostic — speech (Whisper), proteins (AlphaFold2) all went Transformer afterward.' },
  { year:2020, view:'ssm', name:'Linear Transformer',
    paper:'Transformers are RNNs (Katharopoulos et al.)',
    contrib:'φ kernelization + associativity; complexity O(L²)→O(Ld²)',
    highlights:['lin'],
    note:'The founding work of linear attention: Transformers are RNNs. The efficiency line and the recurrence-revival line converge here.' },
  { year:2021, view:'moe', name:'Switch Transformer',
    paper:'Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al.)',
    contrib:'Top-1 sparse routing; trillion-parameter MoE (GShard 2020 pioneered MoE+Transformer)',
    highlights:['router','exp1','exp2','exp3','exp4','wsum'],
    note:'MoE meets Transformer: total parameters decouple from per-token compute — the key to the trillion-parameter era.' },
  { year:2021, view:'ssm', name:'S4',
    paper:'Efficiently Modeling Long Sequences with Structured State Spaces (Gu et al.)',
    contrib:'HiPPO + structured SSM; swept long-sequence benchmarks',
    highlights:['s4'],
    note:'The return of state-space models: discretized continuous systems, trained as convolutions and inferred as RNNs — Mamba\'s direct predecessor.' },
  { year:2021, view:'train', name:'FLAN instruction tuning',
    paper:'Finetuned Language Models are Zero-Shot Learners (Wei et al.)',
    contrib:'instruction-formatted task data → zero-shot generalization to unseen tasks',
    highlights:['sft','pretrain'],
    note:'The foundational work showing instruction tuning works: SFT became standard from then on, directly giving rise to the InstructGPT three-stage paradigm.' },
  { year:2022, view:'train', name:'InstructGPT / ChatGPT',
    paper:'Training language models to follow instructions with human feedback (Ouyang et al.)',
    contrib:'The SFT → RLHF → PPO three-stage paradigm established',
    highlights:['sft','rlhf','pretrain'],
    note:'The 1.3B InstructGPT beat the 175B GPT-3 on human preference — landmark evidence that alignment matters more than scale; ChatGPT launched that November.' },
  { year:2022, view:'train', name:'Chinchilla',
    paper:'Training Compute-Optimal Large Language Models (Hoffmann et al.)',
    contrib:'parameters and data should scale proportionally (70B ↔ 1.4T tokens)',
    highlights:['scaling'],
    note:'Corrected Kaplan-era training recipes: most contemporary models were under-trained on data — directly rewriting the training-budget allocation of every subsequent flagship.' },
  { year:2022, view:'attn', name:'FlashAttention',
    paper:'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al.)',
    contrib:'tiled online softmax: exact attention memory O(L²)→O(L)',
    highlights:['flash'],
    note:'Same math, better memory access — now the industry-default kernel (v2 2023, v3 2024 keep evolving). Long-context training became accessible to everyone.' },
  { year:2022, view:'ssm', name:'RWKV',
    paper:'RWKV: Reinventing RNNs for the Transformer Era (Peng et al.)',
    contrib:'channel-wise decay: an RNN with parallel training and O(1) inference state',
    highlights:['lin','rwkv'],
    note:'Community standard-bearer of the RNN revival: rewriting attention as a linear form that is both parallelizable and recurrent.' },
  { year:2023, view:'dense', name:'LLaMA (the modern Dense paradigm)',
    paper:'LLaMA: Open and Efficient Foundation Language Models (Touvron et al.)',
    contrib:'The RMSNorm + RoPE + SwiGLU trio; ignited the open-source ecosystem',
    highlights:['tok','emb','rope','rms1','swiglu','lnf'],
    note:'The definitive modern Decoder-Only Dense: Pre-Norm/RMSNorm, rotary position embeddings, gated FFN — all later Dense models like Qwen followed.' },
  { year:2023, view:'attn', name:'GQA',
    paper:'GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al.)',
    contrib:'grouped-shared KV: the sweet spot of quality vs cache (standard since LLaMA-2 70B)',
    highlights:['mqa','gqa'],
    note:'MQA-MHA compromise; also convertible from old checkpoints via uptraining. Adopted across Qwen2/3.' },
  { year:2023, view:'ssm', name:'Mamba',
    paper:'Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao)',
    contrib:'selective mechanism + hardware-aware scan; linear complexity rivaling Transformers',
    highlights:['s4','mamba'],
    note:'SSM\'s highlight: input-dependent parameters let the model learn to forget; Mamba-2 (2024) unifies with linear attention under the SSD framework.' },
  { year:2023, view:'moe', name:'Mixtral 8x7B',
    paper:'Mixtral of Experts (Jiang et al.)',
    contrib:'open-source top-2 sparse MoE: 47B total / 13B active params',
    highlights:['router','exp1','exp2','exp3','exp4','wsum'],
    note:'MoE went mainstream open source: at equal inference cost it rivals larger Dense models — most new flagships since have been MoE.' },
  { year:2023, view:'attn', name:'Mistral 7B（SWA）',
    paper:'Mistral 7B (Jiang et al.)',
    contrib:'sliding-window attention: long-document inference cache O(L)→O(w)',
    highlights:['swa','flash'],
    note:'GQA+SWA lets a 7B model handle long documents; StreamingLLM\'s attention sinks explain the first-token attention sink phenomenon.' },
  { year:2023, view:'train', name:'DPO direct preference optimization',
    paper:'Direct Preference Optimization (Rafailov et al.)',
    contrib:'preference alignment without a reward model or RL sampling',
    highlights:['dpo','rlhf'],
    note:'Folds RLHF\'s two steps into one classification-style loss: the open-source alignment barrier dropped sharply; RLHF and DPO have coexisted complementarily ever since.' },
  { year:2024, view:'frontier', name:'TTT layers / memory layers',
    paper:'Learning to (Learn at Test Time) (Sun et al.) · Memory Layers at Scale (Meta)',
    contrib:'hidden state = online learner; sparse key-value memory replaces FFN',
    highlights:['ttt','memlayer'],
    note:'Two memory routes started together: TTT turns the state into a learnable network; Memory Layers externalize factual knowledge into retrievable parameters.' },
  { year:2024, view:'frontier', name:'BLT byte-level LM',
    paper:'Byte Latent Transformer: Patches Scale Better Than Tokens (Meta)',
    contrib:'entropy-driven dynamic patches; drops the BPE tokenizer',
    highlights:['blt'],
    note:'Compute allocated by information entropy: harder-to-predict bytes get more compute; at equal compute it outperforms Llama 3 on robustness — a direct challenge to whether tokenizers are necessary.' },
  { year:2024, view:'gen', name:'MTP multi-token prediction',
    paper:'DeepSeek-V3 Technical Report (DeepSeek-AI)',
    contrib:'auxiliary heads predict several steps at once: denser training signal + speculative drafts',
    highlights:['autoreg','mtp'],
    note:'+1.5~3% on multiple benchmarks; at inference MTP heads serve directly as EAGLE-style draft models — one objective serving both training and inference speedup.' },
  { year:2024, view:'attn', name:'DeepSeek-V2/V3（MLA）',
    paper:'DeepSeek-V2 / DeepSeek-V3 Technical Report (DeepSeek-AI)',
    contrib:'MLA compresses KV cache tens of times + MoE 671B/37B active params',
    highlights:['gqa','mla','flash'],
    note:'Current endpoint of the KV-compression line: cache stores only low-rank latent vectors; V3 adds auxiliary-loss-free balancing and the MTP training objective.' },
  { year:2024, view:'ssm', name:'xLSTM / Mamba-2',
    paper:'xLSTM: Extended Long Short-Term Memory (Beck et al.) · Transformers are SSMs (Dao & Gu)',
    contrib:'exponentially gated matrix memory; SSM and attention unified as SSD',
    highlights:['xlstm','mamba'],
    note:'The LSTM originators\' answer 27 years later; theoretically, linear attention and SSMs converge — the hybrid-architecture era begins.' },
  { year:2025, view:'gen', name:'Diffusion language models',
    paper:'LLaDA: Large Language Diffusion Models · Mercury / Gemini Diffusion (Inception Labs)',
    contrib:'parallel denoising generation directly challenges the autoregressive paradigm',
    highlights:['diffusion','autoreg','spec'],
    note:'LLaDA 8B first validated diffusion LM scaling laws; Mercury focuses on 5-10x generation speed — the first real fork in generation paradigms since 2017.' },
  { year:2025, view:'attn', name:'Gemma 3 (stabilization)',
    paper:'Gemma 3 Technical Report (Google)',
    contrib:'QK-Norm + gated attention + 5:1 local/global + multimodal',
    highlights:['qknorm','swa'],
    note:'The stabilization trio (QK-Norm/soft-cap/z-loss) became the invisible foundation of 100B-scale training — changing no math, just ensuring trainability.' },
  { year:2025, view:'train', name:'DeepSeek-R1 (verifiable-reward RL)',
    paper:'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek-AI)',
    contrib:'pure RL (GRPO) spontaneously emerges long chains of thought and reflection',
    highlights:['rlvr','rlhf','dpo'],
    note:'Reasoning was shown for the first time to grow from RL with verifiable rewards rather than supervised demonstrations alone — a new watershed in training paradigms.' },
  { year:2025, view:'frontier', name:'Titans / latent reasoning',
    paper:'Titans: Learning to Memorize at Test Time (Google) · Reasoning by Latent Space (Huginnet)',
    contrib:'test-time neural memory; deep-looping latent reasoning',
    highlights:['titans','looped','ttt'],
    note:'Thinking moves from output tokens into the architecture: memory keeps learning at test time and depth adapts to difficulty — two routes complementary to o1-style long chains of thought.' },
  { year:2025, view:'moe', name:'Qwen3 / Kimi K2',
    paper:'Qwen3 Technical Report (Alibaba) · Kimi K2: Open Agentic Intelligence (Moonshot AI)',
    contrib:'Fine-grained MoE goes mainstream: Qwen3-235B-A22B (128 choose 8); K2 1T total / 32B active (MLA)',
    highlights:['gqa','router','exp1','exp4'],
    note:'Two answers from current open-source flagships: Qwen3-style dual modes (thinking/non-thinking); K2 compresses active params to 3% via MoE+MLA.' },
  { year:2025, view:'attn', name:'NSA / MoBA sparse attention',
    paper:'Native Sparse Attention (DeepSeek-AI) · MoBA: Mixture of Block Attention (Moonshot AI)',
    contrib:'trainable block-sparse attention; near-linear long-context cost',
    highlights:['flash','sparse','linear'],
    note:'The next stop for exact attention: compute only important blocks with sparsity baked into pre-training — DeepSeek and Moonshot delivered in the same month.' },
  { year:2025, view:'ssm', name:'Kimi Linear（KDA）',
    paper:'Kimi Linear: An Expressive, Efficient Attention Architecture (Moonshot AI)',
    contrib:'gated linear attention in 3/4 of layers + MLA in 1/4; million-token context',
    highlights:['kda','lin','mamba','mla'],
    note:'The flagship hybrid: linear layers for efficiency, full attention for precise retrieval — where the RNN-revival and attention-evolution lines converge.' },
  { year:2026, view:'attn', name:'DeepSeek V4（CSA + mHC）',
    paper:'DeepSeek-V4 Technical Report (DeepSeek-AI)',
    contrib:'compressed sparse attention: 1M-context KV cache down to ~10% of V3.2\'s',
    highlights:['sparse','mla','flash'],
    note:'1.6T total / 49B active params (MIT open source) + mHC hyper-connections improving residual pathways; three inference modes turn thinking depth into an API parameter.' },
  { year:2026, view:'ssm', name:'Kimi K3 (2.8T full-stack overhaul)',
    paper:'Kimi K3: Open Frontier Intelligence (Moonshot AI)',
    contrib:'KDA+MLA hybrid · LatentMoE · Attention Residuals · full-stack NoPE',
    highlights:['kda','mla','lin'],
    note:'First open-source 3T-class model (896 experts choose 16) + first all-NoPE frontier model + native multimodal — the 2.8T productionization of the Kimi Linear paper.' },
  { year:2026, view:'attn', name:'The inference-efficiency arms race',
    paper:'DeepSeek V4 · Kimi K3 · Nemotron 3 · Qwen3.8 (the 2026 open-source flagship wave)',
    contrib:'component-level efficiency swaps: MoE→LatentMoE, attention→MLA+linear hybrid, RoPE→NoPE',
    highlights:['flash','mla','sparse','linear'],
    note:'The 2026 mainstream is no longer stacking parameters but swapping every component for an inference-efficiency-tuned version: release cycles compressed to 6-8 weeks, frontier capabilities fully open-sourced (MIT/Apache).' },
  { year:2025, view:'frontier', name:'Energy-Based Transformers',
    paper:'Energy-Based Transformers are Scalable Learners (2025)',
    contrib:'energy function + test-time energy descent for implicit reasoning',
    highlights:['ebt','ttt'],
    note:'Building deliberation into architecture: reasoning = iterative energy minimization. The newest branch of the test-time-compute line alongside TTT/Titans; conclusions still await large-scale validation.' }
];


/* 注册到全局，供 index.html 主脚本使用 */
window.CONTENT_PACK = { MODULE_DETAILS, HISTORY, ERAS, ARCH_NODES, ARCH_FRAMES, ARCH_EDGES,
  ARCH_SPECIAL_EDGES, ARCH_RESIDUALS, ARCH_VIEW_DAWN, ARCH_VIEW_PRE, ARCH_VIEW_DENSE,
  ARCH_VIEW_MOE, ARCH_VIEW_ATTN, ARCH_VIEW_SSM, ARCH_VIEW_GEN, ARCH_VIEW_FRONTIER,
  ARCH_VIEW_TRAIN, ARCH_VIEWS };
})();
