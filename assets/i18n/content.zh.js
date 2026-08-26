(function(){
/* ============================================================
   内容语言包 · 中文（zh）
   从 index.html 抽出的全部展示型数据：模块详解 / 时间轴 / 架构视图。
   由 index.html 的语言 loader 动态加载 zh 或 en 版本。
   ============================================================ */
const MODULE_DETAILS = {
  'input': {
    title: '输入 Tokens',
    desc: '源序列经分词得到 token 序列，再映射为词表索引。',
    latex: ['x=[x_1,x_2,\\dots,x_n],\\; x_i\\in\\mathbb{Z},\\; x_i<|V|'],
    shapes: ['[seq_len]', '[batch, seq_len, d_model]'],
    code: "const ids = tokens.map(t => Vocab.indexOf(t));\nconst idx = tf.tensor1d(ids, 'int32');",
    params: '0（仅查索引）'
  },
  'embed': {
    title: '输入嵌入层（Embedding）',
    desc: '词表索引映射为 d_model 维向量，并乘 √d_model 放大（原论文做法），随后与位置编码相加。',
    latex: [
      '\\text{Emb}(x)=E[x]\\cdot\\sqrt{d_{model}}',
      'E\\in\\mathbb{R}^{|V|\\times d_{model}}'
    ],
    shapes: ['[batch, seq_len]', '[batch, seq_len, d_model]'],
    code: "const emb = tf.gather(E, ids)\n            .mul(Math.sqrt(dModel))\n            .reshape([1, seqLen, dModel]);",
    params: 'V × d_model'
  },
  'pe': {
    title: '位置编码（Positional Encoding）',
    desc: '正弦/余弦函数为每个位置生成唯一的“时钟指纹”，使模型感知词序；与嵌入逐元素相加。',
    latex: [
      'PE_{(pos,\\,2i)}=\\sin\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)',
      'PE_{(pos,\\,2i+1)}=\\cos\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)'
    ],
    shapes: ['[1, max_len, d_model]', '相加后 [batch, seq_len, d_model]'],
    code: "const denom = Math.pow(10000, (2 * i) / dModel);\npe[pos*dModel+i]   = Math.sin(pos / denom); // 偶数维\npe[pos*dModel+i+1] = Math.cos(pos / denom); // 奇数维\nconst peTensor = tf.tensor3d(pe, [1, maxLen, dModel]);",
    params: '0（固定函数，无可训练参数）'
  },
  'enc-mha': {
    title: '多头自注意力（Multi-Head Self-Attention）',
    desc: '每个位置同时作为 Query/Key/Value，从全序列聚合信息；h 个头在不同子空间并行计算后拼接。',
    latex: [
      '\\text{Attention}(Q,K,V)=\\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
      '\\text{head}_i=\\text{Attention}(QW_i^Q,\\,KW_i^K,\\,VW_i^V)',
      '\\text{MultiHead}(Q,K,V)=\\text{Concat}(\\text{head}_1,\\dots,\\text{head}_h)\\,W^O'
    ],
    shapes: ['[batch, seq_len, d_model]', '中间 [batch, h, seq_len, d_k]', '[batch, seq_len, d_model]'],
    code: "const scores  = tf.matMul(Q, K, false, true).div(Math.sqrt(dk));\nconst weights = tf.softmax(scores, -1);          // [B,H,L,L]\nconst output  = tf.matMul(weights, V);           // 加权求和 V\n// 多头：reshape([B,L,h,dh]).transpose([0,2,1,3]) 拆头，逆操作拼回",
    params: '4·(d² + d)　（W^Q,W^K,W^V,W^O 及偏置）'
  },
  'dec-masked-mha': {
    title: '掩码多头自注意力（Decoder）',
    desc: '解码器只能看见已生成的前缀：上三角位置在 softmax 前被置为 −∞，概率归零，防止“偷看未来”。',
    latex: [
      '\\text{Attention}(Q,K,V)=\\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}+M\\right)V',
      'M_{ij}=\\begin{cases}0 & j\\le i\\\\ -\\infty & j>i\\end{cases}'
    ],
    shapes: ['[batch, Lt, d_model]', '掩码 [1, 1, Lt, Lt]', '[batch, Lt, d_model]'],
    code: "// 下三角 0 / 其余 -1e9 的加性掩码\nconst mask = tf.where(\n  tf.greater(tf.range(0,L).reshape([1,L]),\n             tf.range(0,L).reshape([L,1])),\n  tf.fill([L,L], -1e9), tf.zeros([L,L]));\nconst masked = scores.add(mask.reshape([1,1,L,L]));",
    params: '4·(d² + d)'
  },
  'dec-cross': {
    title: '交叉注意力（Cross-Attention）',
    desc: '解码器的 Encoder-Decoder Attention：Q 来自解码器当前层，K/V 来自编码器最终输出——源句信息的唯一入口（2014 年 Seq2Seq+Attention 思想的继承者）。',
    latex: [
      '\\text{head}_i=\\text{Attention}(H_{dec}W_i^Q,\\,H_{enc}W_i^K,\\,H_{enc}W_i^V)'
    ],
    shapes: ['Q [batch, Lt, d_model]', 'K/V [batch, Ls, d_model]', '输出 [batch, Lt, d_model]'],
    code: "const c = crossMha.forward(hDec, encOut, encOut, null);\n// Q ← 解码器隐状态；K/V ← 编码器输出（同一线性投影族）",
    params: '4·(d² + d)'
  },
  'addnorm': {
    title: '残差连接 + 层归一化（Add & Norm）',
    desc: '残差旁路让梯度直达底层、可堆叠深层网络；LayerNorm 稳定各层激活分布。',
    latex: [
      'h=\\text{LayerNorm}\\big(x+\\text{Sublayer}(x)\\big)',
      '\\text{LN}(x)=\\gamma\\odot\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\varepsilon}}+\\beta'
    ],
    shapes: ['[batch, seq_len, d_model]', '形状不变（逐元素）'],
    code: "const { mean, variance } = tf.moments(x, -1, true);\nconst y = x.sub(mean)\n           .div(variance.add(1e-6).sqrt())\n           .mul(gamma).add(beta);\n// Add & Norm：ln(x.add(sublayerOut))",
    params: '每处 2·d（γ 与 β）；每层共 2 处 → 4·d'
  },
  'ffn': {
    title: '前馈网络（Feed-Forward Network）',
    desc: '逐位置独立的两层 MLP：先扩维到 d_ff（512→2048），ReLU 后再投回 d_model。参数量的大头。',
    latex: [
      '\\text{FFN}(x)=\\max\\!(0,\\,xW_1+b_1)W_2+b_2'
    ],
    shapes: ['[batch, seq_len, d_model]', '[batch, seq_len, d_ff]', '[batch, seq_len, d_model]'],
    code: "const h = tf.relu(x.matMul(W1).add(b1)); // [B,L,d_ff]\nconst out = h.matMul(W2).add(b2);        // [B,L,d_model]",
    params: '2·d·d_ff + f + d'
  },
  'enc-stack': {
    title: 'Encoder 完整结构（× N 层）',
    desc: '上述子层串联为一层，堆叠 N 层（原论文 N=6）。所有位置完全并行，无循环依赖。',
    latex: [
      'H^{(l)}=\\text{LN}\\Big(H^{(l-1)}+\\text{FFN}\\big(\\text{LN}(H^{(l-1)}+\\text{MHA}(H^{(l-1)}))\\big)\\Big)'
    ],
    shapes: ['输入 [batch, seq_len, d_model]', 'N 层间形状不变', '输出 [batch, seq_len, d_model]'],
    code: "for (let l = 0; l < numLayers; l++) {\n  const r = encLayers[l].forward(x); // MHA → Add&Norm → FFN → Add&Norm\n  x = r.output;\n}",
    params: '每层 4·(d²+d) + 2·d·f + f + 5·d，× N 层'
  },
  'dec-stack': {
    title: 'Decoder 完整结构（× N 层）',
    desc: '每层比 Encoder 多一个交叉注意力子块；自注意力带因果掩码，保证第 t 步只依赖 < t 步。',
    latex: [
      'z=\\text{LN}_1(x+\\text{MaskedMHA}(x)),\\; z^{\\prime}=\\text{LN}_2\\big(z+\\text{CrossAttn}(z,\\,H_{enc})\\big)'
    ],
    shapes: ['目标输入 [batch, Lt, d_model]', '交叉注意力 K/V [batch, Ls, d_model]', '输出 [batch, Lt, d_model]'],
    code: "for (const layer of decLayers) {\n  const r = layer.forward(y, encOut, causalMask(Lt));\n  y = r.output;\n}",
    params: '每层 8·(d²+d) + 2·d·f + f + 7·d，× N 层'
  },
  'out': {
    title: '输出投影 + Softmax',
    desc: '解码器末位置隐状态投影到词表维度，softmax 得到下一个 token 的概率分布（逐词生成循环）。注意：本演示权重为随机初始化，分布仅展示机制而非语言能力。',
    latex: [
      'P(u\\,|\\,x)=\\text{softmax}\\big(H_L W_{out}+b\\big),\\quad W_{out}\\in\\mathbb{R}^{d\\times|V|}'
    ],
    shapes: ['[batch, 1, d_model]', '[batch, |V|]', '概率和为 1'],
    code: "const last = y.slice([0, Lt-1, 0], [1, 1, dModel]);\nconst logits = last.matMul(Wout).add(bout); // [1,1,V]\nconst probs  = tf.softmax(logits, -1);",
    params: 'd × V + V'
  },

  /* ============ 前史（RNN → Transformer） ============ */
  'rnn': {
    title: '简单 RNN（Elman, 1990）',
    desc: '最原始的序列模型：隐藏状态沿时间逐步递推，上一步的输出是下一步的输入。理论上能记住任意长的历史，实际上梯度连乘导致只能记住约 10~20 步。',
    latex: [
      'h_t=\\tanh(W_h h_{t-1}+W_x x_t+b),\\quad y_t=W_y h_t',
      '\\frac{\\partial h_t}{\\partial h_0}=\\prod_{k\\le t} W_h^{\\top}\\text{diag}(\\varphi\\prime)\\ \\text{（连乘 → 消失/爆炸）}'
    ],
    shapes: ['[batch, seq, d] 逐步展开','t 时刻必须等 t−1 算完（无法并行）'],
    code: "h = tf.zeros([B, d]);\nfor (const x_t of xs)          // 必须串行\n  h = tf.tanh(h.matMul(Wh).add(x_t.matMul(Wx)));",
    params: 'd² + d·d + d',
    refs: ['Elman 1990 · Finding Structure in Time',
           'Rumelhart et al. 1986 · Learning representations by back-propagating errors']
  },
  'lstm': {
    title: 'LSTM（1997）与 GRU（2014）',
    desc: '给 RNN 装上「传送带」：细胞状态 cₜ 沿时间近似线性流动，三个门（输入/遗忘/输出）控制读写擦除，缓解梯度消失。GRU（2014）合并门结构、参数更少，效果相当。',
    latex: [
      'c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde{c}_t,\\quad h_t=o_t\\odot\\tanh(c_t)',
      'i_t=\\sigma(W_i[h_{t-1},x_t])\\ \\text{（遗忘/输出门同理）}'
    ],
    shapes: ['[batch, seq, 4d]（四门拼接一次算完）','c_t / h_t [batch, d]'],
    code: "const gates = tf.matMul(tf.concat([h, x], 1), W);  // i f o g\nconst c = f.mul(cPrev).add(i.mul(tf.tanh(g)));     // 细胞状态更新\nconst h = o.mul(tf.tanh(c));",
    params: '4·(d·(d+e)+d)（GRU 为 3 组）',
    refs: ['Hochreiter & Schmidhuber 1997 · Long Short-Term Memory',
           'Cho et al. 2014 · Learning Phrase Representations (GRU)']
  },
  's2s': {
    title: 'Seq2Seq（2014）',
    desc: 'Encoder-Decoder 框架：编码器把变长源句压缩为一个固定向量 c，解码器从 c 逐步生成目标句。确立了「编码-解码」范式——但固定向量成为长句的性能瓶颈。',
    latex: [
      'c=\\text{Enc}(x_{1:n}),\\quad P(y)=\\prod_t P(y_t\\mid y_{<t},c)'
    ],
    shapes: ['源句 [n, d] → c [d]（信息瓶颈）','目标句逐步生成'],
    code: "const c = encodeRNN(src).slice([srcLen-1]);  // 末状态 = 整句\nlet h = c;\nfor (let t = 0; t < maxLen; t++) h = decodeRNN(h, yPrev);",
    params: '2 组 RNN 参数',
    refs: ['Sutskever et al. 2014 · Sequence to Sequence Learning with Neural Networks']
  },
  'batt': {
    title: '加性注意力（Bahdanau, 2015）',
    desc: '打破固定向量瓶颈：解码每一步动态回看编码器全部状态，按对齐分数 α 加权求和得到上下文。注意力权重可视化即「软对齐」——2017 自注意力的直系前身（同年 Luong 提出乘性/点积版本）。',
    latex: [
      'e_{ti}=a(s_{t-1},h_i),\\quad \\alpha_{ti}=\\text{softmax}_i(e_{ti})',
      'c_t=\\textstyle\\sum_i \\alpha_{ti} h_i'
    ],
    shapes: ['分数 [L_t, L_s] → 权重 α（行和=1）','上下文 c_t [d]'],
    code: "const e = tf.tanh(sPrev.matMul(Wa).add(H.matMul(Ua)));\nconst alpha = tf.softmax(e.matMul(v), -1);   // 对齐权重\nconst ctx = alpha.matMul(H);                 // 加权上下文",
    params: '对齐网络 ~3·d²',
    refs: ['Bahdanau et al. 2015 · Neural Machine Translation by Jointly Learning to Align and Translate',
           'Luong et al. 2015 · Effective Approaches to Attention-based Neural Machine Translation']
  },
  'selfattn': {
    title: '自注意力（2017，前史的终点）',
    desc: '关键一跃：注意力不再只是「解码器看编码器」，而是让序列自己看自己——Q、K、V 同源。任意两位置一步直连（路径 O(1)），全部位置一次矩阵乘法并行完成，循环被彻底抛弃。',
    latex: [
      '\\text{Attention}(X,X,X)=\\text{softmax}\\!\\left(\\frac{XX^{\\top}}{\\sqrt{d}}\\right)X'
    ],
    shapes: ['[batch, L, d] → [batch, L, d]（一次并行）'],
    code: "const scores = X.matMul(X, false, true).div(Math.sqrt(d));\nreturn tf.softmax(scores, -1).matMul(X);",
    params: '0（不含投影）',
    refs: ['Vaswani et al. 2017 · Attention Is All You Need']
  },

  /* ============ Decoder-Only · Dense 现代组件 ============ */
  'rope': {
    title: 'RoPE 旋转位置编码（RoFormer, 2021）',
    desc: '把位置编码为二维平面旋转：q、k 各自旋转 m·θ、n·θ 后做内积，结果只依赖相对位置 m−n。绝对位置的实现、相对位置的表达，且可外推；配合 YaRN/NTK 插值可扩展到百万级上下文。LLaMA/Qwen/Kimi 全系采用（ALiBi 2022 为线性偏置变体）。',
    latex: [
      '\\begin{pmatrix}q_m^{(2i)}\\\\ q_m^{(2i+1)}\\end{pmatrix}=\\begin{pmatrix}\\cos m\\theta_i & -\\sin m\\theta_i\\\\ \\sin m\\theta_i & \\cos m\\theta_i\\end{pmatrix}\\begin{pmatrix}q^{(2i)}\\\\ q^{(2i+1)}\\end{pmatrix}',
      '\\langle R_m q,\\,R_n k\\rangle=\\langle q,\\,R_{n-m}k\\rangle\\ \\text{（只依赖 }m-n\\text{）}'
    ],
    shapes: ['[batch, seq, d_model] 成对维度旋转','形状不变、0 参数'],
    code: "const theta = Math.pow(10000, -2*i/dModel);\nconst [x, y] = [q[2*i], q[2*i+1]];\nq[2*i]   = x*Math.cos(m*theta) - y*Math.sin(m*theta);\nq[2*i+1] = x*Math.sin(m*theta) + y*Math.cos(m*theta);",
    params: '0（固定旋转，无可训练参数）',
    refs: ['Su et al. 2021 · RoFormer: Enhanced Transformer with Rotary Position Embedding',
           'Press et al. 2022 · ALiBi (Train Short, Test Long)',
           'Peng et al. 2024 · YaRN (上下文扩展)',
           'Moonshot AI 2026 · Kimi K3（首个全栈 NoPE 前沿模型：位置由因果掩码+KDA 状态隐式携带）']
  },
  'rmsnorm': {
    title: 'RMSNorm（2019）+ Pre-Norm',
    desc: 'LayerNorm 的减法：去掉均值中心化，只按均方根缩放，计算更少且效果相当；配合 Pre-Norm（先归一化再进子层）让百层网络稳定训练。GPT-2 起用 Pre-LN，LLaMA 起用 Pre-RMSNorm——2017 的 Post-LN 已成为历史。',
    latex: [
      '\\text{RMSNorm}(x)=\\frac{x}{\\sqrt{\\frac{1}{d}\\textstyle\\sum_i x_i^2+\\varepsilon}}\\odot g',
      '\\text{对比 LayerNorm：无需减去均值 }\\mu(x)'
    ],
    shapes: ['[batch, seq, d_model] → 同形'],
    code: "const ms = tf.mean(tf.square(x), -1, true);\nreturn x.div(ms.add(1e-6).sqrt()).mul(gamma);",
    params: 'd（仅缩放向量 g）',
    refs: ['Zhang & Sennrich 2019 · Root Mean Square Layer Normalization',
           'Xiong et al. 2020 · On Layer Normalization in the Transformer Architecture (Pre-LN)']
  },
  'swiglu': {
    title: 'SwiGLU 门控前馈（2020→LLaMA）',
    desc: '现代 LLM 的 FFN：SiLU 门控分支与线性分支逐元素相乘后再投影，同等参数量下稳定优于 ReLU FFN。因多一个矩阵，d_ff 常取约 ⅔·4d 保持总参数与原设计对齐。激活函数演进线：ReLU(2017) → GELU(GPT/BERT) → SiLU/SwiGLU(LLaMA 起)。',
    latex: [
      '\\text{SwiGLU}(x)=\\big(\\text{SiLU}(xW_{g})\\odot xW_{u}\\big)W_{d},\\quad \\text{SiLU}(x)=x\\cdot\\sigma(x)'
    ],
    shapes: ['[B,L,d] → [B,L,d_ff] → [B,L,d]'],
    code: "const gate = tf.silu(x.matMul(Wg));\nconst up   = x.matMul(Wu);\nreturn gate.mul(up).matMul(Wd);",
    params: '3·d·d_ff',
    refs: ['Shazeer 2020 · GLU Variants Improve Transformer',
           'Touvron et al. 2023 · LLaMA（首次大规模采用）']
  },
  'gqa': {
    title: 'GQA 分组查询注意力（2023）',
    desc: 'Q 头分组共享 KV 头（如 32 个 Q 头共享 8 组 KV）：KV Cache 与解码带宽直接 ÷4，质量接近 MHA；还可从训练好的 MHA 检查点「Uptraining」低成本转换。MQA（2019）是 h_KV=1 的极端特例。Qwen2/3、LLaMA-2/3 的标配。',
    latex: [
      '\\text{head}_i=\\text{Attention}\\big(QW_i^Q,\\,KW_{g(i)}^K,\\,VW_{g(i)}^V\\big),\\quad g(i)=\\big\\lfloor i/(h_Q/h_{KV})\\big\\rfloor'
    ],
    shapes: ['Q [B, 32, L, dh]','K/V [B, 8, L, dh] → tile 广播','KV Cache ÷ 4'],
    code: "const K8  = K.reshape([B,L,8,dh]).transpose([0,2,1,3]);\nconst K32 = tf.tile(K8, [1, 4, 1, 1]);   // 8 组 KV → 32 头\nconst out = sdpa(Q, K32, V32, causalMask(L));",
    params: 'KV 投影省 (1 − h_KV/h_Q)·2·d·d',
    refs: ['Ainslie et al. 2023 · GQA: Training Generalized Multi-Query Transformer Models',
           'Kwon et al. 2023 · vLLM/PagedAttention（Cache 显存管理）']
  },
  'lm-head': {
    title: 'LM Head 输出层',
    desc: 'Final RMSNorm 后的隐状态经 Linear 投影到词表并 softmax，取 argmax/采样作为下一个 token。现代 LLM 常与输入 Embedding 共享权重（权重绑定），省下 V×d 参数；解码侧配合 KV Cache 逐 token 生成。',
    latex: [
      'P(u\\,|\\,x)=\\text{softmax}(h_L W_{out}^{\\top}),\\quad W_{out}=E^{\\top}\\ \\text{（可选绑定）}'
    ],
    shapes: ['[B,L,d] → [B,L,|V|] → 下一个 token'],
    code: "const logits = h.matMul(E.transpose());  // 权重绑定\nconst next = tf.multinomial(tf.softmax(logits.slice([0,-1])), 1);",
    params: 'd×V（绑定后 0 新增）',
    refs: ['Press & Wolf 2017 · Using the Output Embedding to Improve Language Models']
  },

  /* ============ Attention 演进 ============ */
  'mha-ev': {
    title: 'MHA 多头注意力（2017 基线）',
    desc: '每个 Q 头独立配一组 K/V 头：表达力最强，但推理时 KV Cache 随头数线性增长（2·L·h·dₕ/层），长上下文与大批量并发的瓶颈起点——后续所有变体都在回答「KV 能不能更小」。',
    latex: ['\\text{Cache}=2\\,L\\,h\\,d_h\\ \\text{floats / 层 / 序列}'],
    shapes: ['Q/K/V [B, h, L, dh]（h 组独立 KV）'],
    code: "// h 组 KV 全部进 Cache\nconst K = splitHeads(x.matMul(WK));  // [B,h,L,dh]",
    params: '4·(d²+d)',
    refs: ['Vaswani et al. 2017 · Attention Is All You Need']
  },
  'mqa': {
    title: 'MQA 多查询注意力（2019）',
    desc: '所有 Q 头共享同一组 K/V：KV Cache 骤降 h 倍、解码吞吐大幅提升；但表达力损失明显。作为 GQA 的极端特例（h_KV=1）被记入谱系。',
    latex: ['h_{KV}=1:\\; \\text{Cache} \\div h'],
    shapes: ['Q [B,h,L,dh]','K/V [B,1,L,dh]（全头共享）'],
    code: "const K1 = K.reshape([B,L,1,dh]);\nconst Kt = tf.tile(K1, [1, h, 1, 1]);",
    params: 'KV 投影仅 2·d·dh',
    refs: ['Shazeer 2019 · Fast Transformer Decoding: One Write-Head is All You Need']
  },
  'mla': {
    title: 'MLA 多头潜在注意力（DeepSeek-V2/V3、Kimi K2）',
    desc: '把 KV 低秩压缩进潜在向量 c（d_c ≪ d·h），推理时 Cache 只存 c，按需用吸收矩阵升维还原 K/V：显存缩小数十倍且质量不掉；配合 MoE 支撑 671B/1T 级模型的高效推理。',
    latex: [
      'c_{KV}=W_{DKV}x\\in\\mathbb{R}^{d_c},\\quad K=W_{UK}c_{KV},\\; V=W_{UV}c_{KV}',
      '\\text{Cache}: 2Lhd_h \\rightarrow L d_c\\ (\\div \\text{数十倍})'
    ],
    shapes: ['x [B,L,d] → c_KV [B,L,d_c≈512]','按需还原 K/V [B,h,L,dh]'],
    code: "const c = x.matMul(W_DKV);              // 唯一进 Cache 的\nconst K = c.matMul(W_UK), V = c.matMul(W_UV); // 解码侧升维",
    params: '压缩/升维 ~2·d·d_c + 2·d_c·d',
    refs: ['DeepSeek-AI 2024 · DeepSeek-V2 Technical Report (MLA)',
           'DeepSeek-AI 2024 · DeepSeek-V3 Technical Report',
           'Moonshot AI 2025 · Kimi K2（沿用 MLA）']
  },
  'flash': {
    title: 'FlashAttention（2022，IO 感知的精确注意力）',
    desc: '不改数学、只改访存：Q/K/V 分块装入 SRAM，块内在线 softmax（维护运行最大值 m 与和 ℓ），L×L 矩阵从不写回 HBM。长上下文训练提速 2-4×、显存 O(L²)→O(L)，是现代框架默认内核（v2/2023 换并行轴、v3/2024 上 FP8）。',
    latex: [
      'm\\leftarrow\\max(m,\\,m_{\\text{tile}}),\\quad \\ell\\leftarrow e^{m_{old}-m}\\,\\ell+\\textstyle\\sum_{\\text{tile}}e^{s-m}'
    ],
    shapes: ['HBM 读写：O(L²) → O(Ld)','结果与标准 softmax 完全一致'],
    code: "for (const [Kj, Vj] of tiles(K, V)) {   // K/V 分块流入\n  const S  = Qtile.matMul(Kj.T).div(Math.sqrt(dk));\n  mNew = rowmax(m, S); l = exp(m−mNew)*l + rowsum(exp(S−mNew));\n  acc  = exp(m−mNew)*acc + exp(S−mNew).matMul(Vj); m = mNew;\n}\nconst O = acc.div(l);",
    params: '0（内核级优化，非新参数）',
    refs: ['Dao et al. 2022 · FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
           'Dao 2023 · FlashAttention-2', 'Shah et al. 2024 · FlashAttention-3'],
    extra: '<svg width="330" height="168" role="img" aria-label="FlashAttention 分块在线 softmax 示意" style="background:var(--c-code-bg);border-radius:8px;padding:6px;">' +
      '<text x="14" y="18" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">Q×Kᵀ 分块（因果）</text>' +
      (() => { let s=''; for(let i=0;i<4;i++)for(let j=0;j<4;j++){ const on=j<=i;
        s+='<rect x="'+(14+j*26)+'" y="'+(26+i*22)+'" width="22" height="18" rx="3" fill="'+
        (on?'rgba(126,166,255,.75)':'rgba(160,175,200,.18)')+'"/>'; }
        s+='<rect x="12" y="24" width="106" height="90" rx="6" fill="none" stroke="var(--c-accent)" stroke-width="1.6" stroke-dasharray="4 3"/>';
        return s; })() +
      '<text x="140" y="40" font-size="12" fill="var(--c-text)" font-family="var(--font-mono)">SRAM 常驻：</text>' +
      '<text x="140" y="58" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">Q 分块 + m, ℓ, O 累计器</text>' +
      '<text x="140" y="84" font-size="12" fill="var(--c-text)" font-family="var(--font-mono)">HBM：</text>' +
      '<text x="140" y="102" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">从不物化 L×L 矩阵</text>' +
      '<path d="M 14 132 L 118 132" stroke="var(--c-accent)" stroke-width="1.6" marker-end="url(#arrow)"/>' +
      '<text x="14" y="152" font-size="11.5" fill="var(--c-text-dim)" font-family="var(--font-mono)">K/V 分块依次流入 → 在线 softmax 修正</text>' +
      '</svg>'
  },
  'sparse': {
    title: '可训练稀疏注意力（NSA / MoBA, 2025）',
    desc: '注意力矩阵大部分是低值噪声：只对重要块算精确注意力，其余跳过或压缩。DeepSeek NSA 把稀疏模式做进预训练（硬件对齐的块稀疏核）；Kimi MoBA 以「块级 Top-K 门控」混合全/稀疏注意力，长上下文成本近线性。',
    latex: [
      'O=\\text{Attn}(Q,\\,\\text{TopK-blocks}(K,V)),\\quad g=\\text{softmax}(\\text{gate})'
    ],
    shapes: ['L×L → 每行只保留 ~k 个块','复杂度 ~O(L·k·w)'],
    code: "// MoBA：块级门控选 Top-K 块\nconst blockScores = qBlock.matMul(kBlocks.T);\nconst topK = tf.topk(blockScores, k);  // 其余块不计算",
    params: '0~少量门控参数',
    refs: ['DeepSeek-AI 2025 · Native Sparse Attention (NSA)',
           'Moonshot AI 2025 · MoBA: Mixture of Block Attention for Long-Context LLMs',
           'DeepSeek-AI 2025 · V3.2-Exp（DSA 稀疏注意力，NSA 工程化）']
  },
  'hybrid': {
    title: '混合架构（少数全注意力 + 多数线性层）',
    desc: '全注意力负责精确检索、线性层负责廉价序列建模，按比例（常 1:3 ~ 1:7）混搭：Jamba（AI21，Mamba+Attention）、Griffin/Hawk（DeepMind，门控线性递归）、Samba（Microsoft）、Kimi Linear（KDA+MLA）——长上下文效率与质量的新平衡点。',
    latex: ['\\text{层序列}=[\\underbrace{\\text{Linear}}_{3/4},\\dots,\\underbrace{\\text{Full}}_{1/4}]\\times N'],
    shapes: ['同 Decoder Block，逐层替换注意力类型'],
    code: "for (l of layers) x = l.isFull ? fullAttn(x) : linearAttn(x);",
    params: '介于 Dense 与 Linear 之间',
    refs: ['Lieber et al. 2024 · Jamba: A Hybrid Transformer-Mamba Language Model',
           'De et al. 2024 · Griffin: Mixing Gated Linear Recurrences with Local Attention',
           'Moonshot AI 2025 · Kimi Linear']
  },

  /* ============ 线性 / SSM / RNN 复兴 ============ */
  'linear': {
    title: '线性注意力（Katharopoulos, 2020）',
    desc: '用核函数 φ 改写 softmax 注意力并利用结合律：先算 φ(K)ᵀV（一个 d×d 矩阵），复杂度 O(L·d²) 摆脱 L²；同一公式可改写为状态递推 Sₜ=Sₜ₋₁+φ(kₜ)vₜᵀ——「Transformer 就是 RNN」。后续 RWKV/xLSTM/KDA 都在这条线上做门控与衰减改进。',
    latex: [
      '\\text{Attn}(Q,K,V)=\\frac{\\phi(Q)\\big(\\phi(K)^{\\top}V\\big)}{\\phi(Q)\\phi(K)^{\\top}\\mathbf{1}}',
      'S_t=S_{t-1}+\\phi(k_t)v_t^{\\top},\\quad u_t=\\frac{S_t\\phi(q_t)}{\\phi(q_t)^{\\top}z_t}'
    ],
    shapes: ['φ(K)ᵀV：[d,d] 常量状态','每步 O(d²)，与 L 无关'],
    code: "const KV = phi(K).transpose().matMul(V);  // [d,d] 先算这个\nconst out = phi(Q).matMul(KV);\n// 流式推理：S = S.add(phi(k_t).outer(v_t));",
    params: '0（φ 固定特征映射）',
    refs: ['Katharopoulos et al. 2020 · Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention']
  },
  'rwkv': {
    title: 'RWKV（2022）',
    desc: '把注意力改写为带通道级时间衰减的线性 RNN：训练时可并行（类 Transformer），推理时 O(1) 状态（类 RNN）。开源社区长青的线性模型线。',
    latex: [
      'wkv_t=\\frac{\\sum_i e^{-(t-1-i)w+k_i}v_i+e^{u+k_t}v_t}{\\sum_i e^{-(t-1-i)w+k_i}+e^{u+k_t}}'
    ],
    shapes: ['每通道标量衰减 w','状态 [d] 级别'],
    code: "// 通道级衰减递推（数值稳定版省略）\na = a.mul(wDecay).add(v_t); b = b.mul(wDecay).add(1);",
    params: '~同 Transformer Block',
    refs: ['Peng et al. 2023 · RWKV: Reinventing RNNs for the Transformer Era']
  },
  'xlstm': {
    title: 'xLSTM（2024，LSTM 复兴）',
    desc: 'LSTM 原作者团队 27 年后的回应：sLSTM 用指数门控+新记忆混合，mLSTM 用矩阵记忆 + 协方差更新（可并行）。与 Mamba/RWKV 同属「RNN 复兴」浪潮。',
    latex: ['c_t=f_t\\odot c_{t-1}+i_t\\,v_t,\\quad C_t=C_{t-1}\\odot f_t+n_t v_t^{\\top}\\ \\text{(mLSTM 矩阵记忆)}'],
    shapes: ['记忆由向量 [d] 扩展为矩阵 [d,d]'],
    code: "C = C.mul(f).add(n.outer(v));  // 矩阵细胞状态\nh = C.matMul(q).div(norm);",
    params: '~1.5× LSTM',
    refs: ['Beck et al. 2024 · xLSTM: Extended Long Short-Term Memory']
  },
  's4': {
    title: 'S4 结构化状态空间（2021）',
    desc: '连续状态空间系统经 HiPPO 初始化 + 结构化对角化，可化为长卷积并行训练、递推推理：LRA 长序列基准横扫，是 Mamba 的直接前身。',
    latex: [
      "h'(t)=\\bar{A}h(t)+\\bar{B}x(t),\\quad y=\\bar{C}h(t)\\ \\text{（离散化 SSM）}"
    ],
    shapes: ['核 K = CĀᵏB̄：长度 L 的卷积核'],
    code: "const K = ssmConvKernel(A, B, C, L);  // 长卷积形式\nconst y = conv1d(x, K);               // 训练并行\n// 推理：h = A.mul(h).add(B.mul(x));",
    params: 'A/B/C 结构化参数 ~O(d·N)',
    refs: ['Gu et al. 2021 · Efficiently Modeling Long Sequences with Structured State Spaces (S4)']
  },
  'mamba': {
    title: 'Mamba 选择性 SSM（2023）',
    desc: '让 SSM 的 Δ/B/C 依赖输入（选择机制）：模型可以按内容遗忘或记住信息，弥补 SSM「一视同仁」的缺陷；配合硬件感知并行扫描，训练快 5×、推理 O(1) 状态。Mamba-2（2024）证明其与线性注意力的 SSD 统一框架。',
    latex: [
      'h_t=\\bar{A}(x_t)h_{t-1}+\\bar{B}(x_t)x_t,\\quad y_t=\\bar{C}(x_t)h_t\\ \\text{（参数随输入变化）}'
    ],
    shapes: ['状态 [B,d,N]','扫描 O(L·d·N)'],
    code: "for (const x_t of xs) {            // 硬件感知扫描（并行前缀和）\n  h = Abar(x_t).mul(h).add(Bbar(x_t).mul(x_t));\n  y_t = Cbar(x_t).dot(h);\n}",
    params: 'Δ/B/C 投影 ~3·d·d',
    refs: ['Gu & Dao 2023 · Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
           'Dao & Gu 2024 · Transformers are SSMs (Mamba-2/SSD)']
  },
  'kda': {
    title: 'Kimi Linear · KDA（2025）',
    desc: 'Moonshot 的混合线性架构：KDA（Kimi Delta Attention，带门控增量记忆的线性注意力，可精细遗忘）占 3/4 层，MLA 全注意力占 1/4 层。3B/48B-A3B 规模上全面对标同尺寸全注意力模型：百万级上下文、显存省 ~75%、解码吞吐 ~6×。',
    latex: [
      'S_t=\\alpha_t\\odot S_{t-1}+\\beta_t\\,\\phi(k_t)v_t^{\\top}\\ \\text{（门控增量规则）}'
    ],
    shapes: ['状态 [d,d] 矩阵记忆','层配比 Linear : Full = 3 : 1'],
    code: "for (l of layers)\n  x = (l % 4 === 0) ? mlaBlock(x) : kdaBlock(x);  // 1/4 全注意力",
    params: '~同 Dense Block（线性层更省 KV）',
    refs: ['Moonshot AI 2025 · Kimi Linear: An Expressive, Efficient Attention Architecture']
  },

  /* ============ MoE ============ */
  'moe-router': {
    title: 'MoE 门控路由（Router / Gate）',
    desc: '每个 token 由路由器打分，仅 Top-K（常 K=8）专家被激活：总参数与单 token 计算解耦。谱系：GShard(2020) → Switch(2021, Top-1) → Mixtral(2023, Top-2) → DeepSeek-V3(256 选 8 + 无辅助损失均衡 + 共享专家) → Qwen3(128 选 8) → Kimi K2(384 选 8)。',
    latex: [
      'g=\\text{softmax}\\big(\\text{TopK}(xW_g,\\,K)\\big),\\quad y=\\sum_{i\\in\\text{TopK}} g_i\\,E_i(x)',
      '\\text{DeepSeek-V3：偏置动态调节替代辅助损失（}\\mathcal{L}_{aux}\\text{-free 均衡）}'
    ],
    shapes: ['x [B,L,d] → logits [B,L,E]','Top-K 掩码 → 仅 K/E 专家计算'],
    code: "const logits = x.matMul(Wg);                 // [B,L,E]\nconst {values, indices} = tf.topk(logits, K);\nconst gate = tf.softmax(values, -1);          // 稀疏门控\nlet y = tf.zerosLike(x);\nfor (let k = 0; k < K; k++)\n  y = y.add(experts[indices[k]](x).mul(gate.slice(k)));",
    params: 'd×E + E（路由矩阵）',
    refs: ['Shazeer et al. 2017 · Outrageously Large Neural Networks (Sparsely-Gated MoE)',
           'Lepikhin et al. 2020 · GShard', 'Fedus et al. 2021 · Switch Transformers',
           'DeepSeek-AI 2024 · Auxiliary-Loss-Free Load Balancing']
  },
  'moe-expert': {
    title: 'MoE 专家（Expert = SwiGLU FFN）',
    desc: '每个专家就是一个 SwiGLU FFN。细粒度专家（更多更小的专家）+ 共享专家（常驻处理通用知识，DeepSeek/Qwen3 采用）是当前主流配置；K2 进一步用 MLA 注意力 + 384 专家做到 1T 总参 / 32B 激活。',
    latex: ['E_i(x)=\\text{SwiGLU}_i(x),\\quad \\text{激活参数} \\approx K/E'],
    shapes: ['[B,L,d] → [B,L,d_ff] → [B,L,d]（仅被路由 token）'],
    code: "class Expert {  // = SwiGLU FFN\n  forward(x){ return tf.silu(x.matMul(Wg)).mul(x.matMul(Wu)).matMul(Wd); }\n}",
    params: '每专家 3·d·d_ff（总参数 ×E，激活 ×K/E）',
    refs: ['Jiang et al. 2024 · Mixtral of Experts',
           'Moonshot AI 2025 · Kimi K2: Open Agentic Intelligence']
  },
  'moe-sum': {
    title: '稀疏加权合成',
    desc: '只有被选中的 K 个专家输出参与合成，其余专家本次完全不计算——这就是「万亿总参、百亿激活」的稀疏本质。推理时专家可分布到不同显卡（专家并行 EP）。',
    latex: ['y=\\sum_{i\\in\\text{TopK}} g_i E_i(x)'],
    shapes: ['[B,L,d]（与 Dense FFN 输出同形，直接残差）'],
    code: "y = g1*E1(x) + g2*E2(x);  // 其余 E-2 个专家跳过",
    params: '0',
    refs: ['Fedus et al. 2022 · A Review of Sparse Expert Models in Deep Learning']
  },

  /* ============ Attention 路线 C：长上下文与稳定化 ============ */
  'swa': {
    title: '滑动窗口注意力（SWA）与局部/全局交错',
    desc: '每个位置只看最近 w 个 token：L×L → L×w，推理 Cache 恒为窗口大小。Mistral 首创；Gemma 2/3 用「局部层:全局层 ≈ 5:1」交错——局部层管邻近细节、稀疏的全局层管长程检索。首 token 吸引力现象由 attention sinks（StreamingLLM）解释并利用。',
    latex: [
      'A_{ij}=\\text{softmax}\\!\\left(\\frac{q_i k_j^{\\top}}{\\sqrt{d}}\\right),\\quad j\\in(i-w,\\,i]',
      '\\text{Cache}: O(L)\\rightarrow O(w),\\quad \\text{层配比 局部:全局}=5{:}1\\ \\text{(Gemma 3)}'
    ],
    shapes: ['注意力矩阵 [L,L] → 带状 [L,w]','KV Cache 恒定 w'],
    code: "const start = Math.max(0, t - W);\nconst s = q[t].matMul(K.slice([start, t-start+W]), false, true);\n// 仅窗口内 softmax，窗口外不计算",
    params: '0',
    refs: ['Jiang et al. 2023 · Mistral 7B', 'Riviere et al. 2024 · Gemma 2',
           'Google 2025 · Gemma 3 Technical Report',
           'Xiao et al. 2023 · Efficient Streaming Language Models (Attention Sinks)']
  },
  'ctx-ext': {
    title: '上下文扩展：PI → NTK-aware → YaRN',
    desc: 'RoPE 模型把 4K 训练长度扩到百万级推理的三代方案：PI 线性插值（位置 ÷s）简单有效但高频信息受损；NTK-aware 高频外推、低频内插；YaRN 按波长分频插值 + 注意力温度补偿，10 倍扩展仅需 ~1000 步微调。CodeLlama-100K、Qwen-1M 的底层技术。',
    latex: [
      '\\text{PI}: \\theta_m\\rightarrow\\theta_m/s;\\quad \\text{YaRN}: \\text{按波长 } r(\\theta) \\text{ 分频插值} + \\sqrt{1/t}\\ \\text{温度}'
    ],
    shapes: ['有效上下文 ×10~250（4K→1M）','微调成本 ~1000 步'],
    code: "// YaRN：对第 i 个频率按波长混合插值\nconst r = 2*Math.PI/theta[i];          // 波长\nconst scale = r < base ? 1 : Math.max(0, (r-base)/(L_train-base)) * (1-1/s) + 1/s;",
    params: '0',
    refs: ['Chen et al. 2023 · Extending Context Window via Position Interpolation',
           'bloc97 2023 · NTK-Aware Scaled RoPE', 'Peng et al. 2024 · YaRN']
  },
  'qknorm': {
    title: '训练稳定化三件套：QK-Norm · soft-capping · z-loss',
    desc: '规模越大越容易训崩的主流对策：QK-Norm 对 Q/K 按头做 RMSNorm，防注意力 logits 爆炸（Gemma 3、Chameleon）；logit soft-capping 用 tanh 限幅防输出过自信（Gemma 2）；z-loss 惩罚 softmax 分母漂移（ST-MoE/PaLM）。与 Pre-Norm/RMSNorm 同属「稳定化演进线」。',
    latex: [
      '\\hat{q}=\\text{RMSNorm}(q)\\cdot\\sqrt{d_h};\\quad \\hat{\\ell}=c\\cdot\\tanh(\\ell/c)\\ \\text{(soft-cap)}',
      '\\mathcal{L}_z=\\frac{1}{B}\\sum_b\\big(\\log Z_b\\big)^2\\ \\text{(z-loss)}'
    ],
    shapes: ['注意力 logits 方差受控','训练 loss 曲线平滑'],
    code: "const qn = rmsNorm(Q).mul(Math.sqrt(dh));\nconst kn = rmsNorm(K).mul(Math.sqrt(dh));\nconst logits = softCap(qn.matMul(kn.T), 30); // c=30 tanh 限幅",
    params: 'QK-Norm：4·dₕ/头',
    refs: ['Riviere et al. 2024 · Gemma 2', 'Google 2025 · Gemma 3',
           'Chameleon Team 2024 · Chameleon Mixed-Modal Early-Fusion',
           'Zoph et al. 2022 · ST-MoE (z-loss)']
  },
  'diff': {
    title: '差分注意力（DIFF Transformer, 2024）',
    desc: '仿差分放大器：两组独立 softmax 注意力相减，共模噪声（无关 token 的均匀注意力）被抵消——抗幻觉、长上下文鲁棒、还能省一半头数（每组 h/2 头）。65B 规模验证优于同尺寸 Transformer。',
    latex: [
      '\\text{DiffAttn}(Q)=\\big(\\text{softmax}(Q_1K_1^{\\top})V_1-\\lambda\\,\\text{softmax}(Q_2K_2^{\\top})V_2\\big),\\ \\lambda\\in(0,1)'
    ],
    shapes: ['每组 h/2 头 ×2 组','输出同 MHA 形状'],
    code: "const A1 = softmax(Q1.matMul(K1.T)); const A2 = softmax(Q2.matMul(K2.T));\nreturn A1.sub(A2.mul(lambda)).matMul(V);",
    params: '~同 MHA（两组半宽投影）',
    refs: ['Ye et al. 2024 · Differential Transformer']
  },

  /* ============ 生成范式 ============ */
  'autoreg': {
    title: '自回归下一词预测（NTP）',
    desc: 'LLM 的第一性原理：因果掩码 + 交叉熵，逐 token 左到右。GPT 系（GPT/Qwen/LLaMA/Kimi/DeepSeek）全部采用；KV Cache 让每步生成 O(L)。理解「为什么自回归赢」= 理解简单目标 + 规模的涌现。',
    latex: [
      'P(y)=\\prod_t P(y_t\\mid y_{<t}),\\quad \\mathcal{L}=-\\sum_t\\log P(y_t\\mid y_{<t})'
    ],
    shapes: ['logits [B,L,|V|]','每步生成 O(L)（有 Cache）'],
    code: "const logits = model(yPrev);\nconst loss = tf.losses.softmaxCrossEntropy(oneHot(yNext), logits);",
    params: '0（目标函数，非参数）',
    refs: ['Radford et al. 2018/2019 · GPT-1/2', 'Brown et al. 2020 · GPT-3']
  },
  'mtp': {
    title: 'MTP 多 token 预测（DeepSeek-V3）',
    desc: '在下一词之外再挂 D 个轻量头，一次预测未来 1..D 个 token：训练信号更密集、迫使模型「规划」未来；V3 实测多基准 +1.5~3%。推理时这些头可直接充当投机解码的草稿（EAGLE 思路），一鱼两吃。',
    latex: [
      'p_t^{(k)}=\\text{head}_k(h_t),\\quad \\mathcal{L}=\\sum_{k=1}^{D}\\mathbb{E}\\big[-\\log p_{t+k}^{(k)}\\big]'
    ],
    shapes: ['共享主干 + D 个投影头','每头 [B,L,|V|]'],
    code: "for (let k = 1; k <= D; k++)\n  loss += ce(heads[k](h.slice([0, 0], [-1, L-k])), y.slice([0, k]));",
    params: 'D·d·|V|（辅助头）',
    refs: ['DeepSeek-AI 2024 · DeepSeek-V3 Technical Report (MTP)',
           'Gloeckle et al. 2024 · Better & Faster Large Language Models via Multi-token Prediction',
           'Li et al. 2024 · EAGLE-2']
  },
  'diffusion': {
    title: '扩散语言模型（LLaDA · Mercury · Gemini Diffusion）',
    desc: '生成 = 从全 [MASK] 出发迭代去噪：每轮并行预测所有位置的一个子集，打破左到右依赖，天然可并行、可回改、可控编辑。LLaDA 8B 首次验证扩散 LM 的规模法则（可比肩 LLaMA3-8B）；Mercury/Gemini Diffusion 主打 5-10× 生成速度。当前短板：变长序列、KV Cache 失效、顶级质量仍属自回归。',
    latex: [
      'x_T=\\text{[MASK]}^L\\ \\xrightarrow{\\ p_\\theta(y_t\\mid x_t)\\ }\\ x_0=y\\quad\\text{(逐位置并行去噪)}'
    ],
    shapes: ['每步并行预测全部位置','生成速度 5-10× AR'],
    code: "let x = maskAll(L);\nfor (let step = T; step > 0; step--)\n  x = denoise(x, model, step);   // 每步揭示一部分 token",
    params: '~同规模 AR 模型',
    refs: ['Nie et al. 2025 · LLaDA: Large Language Diffusion Models',
           'Inception Labs 2025 · Mercury Diffusion LLM',
           'Google DeepMind 2025 · Gemini Diffusion']
  },
  'spec': {
    title: '投机解码（Speculative Decoding · EAGLE）',
    desc: '推理加速而非新模型：小草稿模型一次猜 k 个 token，大模型一次前向并行验证，拒绝采样保证输出分布与逐 token 生成完全一致——数学无损的 2-4× 提速。EAGLE 系用目标模型特征层 + MTP 式头做草稿，是当前工业默认。注意边界：投机解码不改结构/参数/训练目标，属「推理系统层」——与 KV Cache 管理（PagedAttention）、continuous batching、量化同层；架构决定一次前向算什么，它决定调用几次前向。',
    latex: [
      '\\text{接受率 }\\alpha=\\mathbb{E}\\big[\\min(1,p/q)\\big],\\quad \\text{期望加速}\\approx\\frac{1-\\alpha^{k+1}}{(1-\\alpha)}\\ \\text{倍}'
    ],
    shapes: ['草稿 k token → 1 次并行验证','每步产出 ≥1 token','模型本身零改动'],
    code: "const draft = smallModel.kTokens(prefix, k);   // 猜 k 个\nconst p = targetModel.probs(prefix.concat(draft));\nconst out = rejectSample(draft, p);            // 无损验证",
    params: '草稿模型 ~1/10 参数（系统级额外开销）',
    refs: ['Leviathan et al. 2023 · Fast Inference from Transformers via Speculative Decoding',
           'Chen et al. 2023 · Accelerating LLM Decoding with Speculative Sampling',
           'Li et al. 2024 · EAGLE-2',
           'Kwon et al. 2023 · vLLM/PagedAttention（同属推理系统层）',
           'Yu et al. 2022 · Orca（continuous batching，同属推理系统层）']
  },

  /* ============ 前沿探索 ============ */
  'blt': {
    title: 'Byte Latent Transformer（BLT, Meta 2024）',
    desc: '去掉 BPE 分词器：字节流按「下一个字节的熵」动态聚成 patch——越可预测处 patch 越大，计算量随信息熵分配。等训练算力下鲁棒性优于 Llama 3（抗噪、无 OOV、跨语言公平），代价是长 patch 依赖局部字节小模型。与 MTP/SWA 同属「改输入分配」的效率线。',
    latex: [
      '\\text{patch 边界} \\approx \\text{熵率突增处},\\quad \\text{计算} \\propto \\text{信息熵}'
    ],
    shapes: ['字节 [B] → 动态 patch [P] → 全局隐状态','P 随内容变化'],
    code: "const patches = entropyPatcher(bytes, smallModel); // 动态边界\nconst h = globalModel(patches);\nconst bytesOut = localDecoder(h, bytes);",
    params: 'patcher(小) + 全局模型 + 局部解码器',
    refs: ['Pagnoni et al. 2024 · Byte Latent Transformer: Patches Scale Better Than Tokens']
  },
  'memlayer': {
    title: '记忆层（Memory Layers at Scale, Meta 2024）',
    desc: '用可训练的稀疏键值记忆（百万级 embedding，product-key 检索 top-k）替换部分 FFN：事实性问答大幅提升而算力开销很小——「知识存记忆、推理靠注意力」的分工假说的工程验证。128M 记忆在 1.3B 模型上 trivia 性能 +100%+。',
    latex: [
      'y=\\textstyle\\sum_{i\\in\\text{topk}(qK)} v_i\\quad\\text{(product-key 量化检索)}'
    ],
    shapes: ['查询 [d] → top-k 记忆槽','稀疏激活 ~k/百万'],
    code: "const keys1 = q.matMul(PK1); keys2 = q.matMul(PK2);\nconst topk = topkProduct(keys1, keys2, k);   // 分解检索\nconst y = V.gather(topk).mean(0);",
    params: '记忆表 1M~1B×d（稀疏访问）',
    refs: ['Berges et al. 2024 · Memory Layers at Scale (Meta)']
  },
  'ttt': {
    title: 'TTT 层：测试时训练（2024）',
    desc: '把 RNN 的「隐藏状态」替换为一个可在线学习的神经网络：每个 token 到来时对隐模型做一步梯度下降，输出 = 更新后模型的前向。序列建模即测试时训练——状态容量从固定向量升级为整个网络。',
    latex: [
      'W_t=W_{t-1}-\\eta\\,\\nabla\\,\\ell\\big(W_{t-1};x_t\\big),\\quad y_t=f\\big(W_t;x_t\\big)'
    ],
    shapes: ['状态 = 权重 W [参数量级]','每 token 一步 SGD'],
    code: "for (const x_t of xs) {\n  const g = grad(w => reconLoss(w, x_t));   // 自监督\n  W = W.sub(g.mul(eta));                    // 状态更新\n  y_t = f(W, x_t);\n}",
    params: '隐模型 ~MLP 级',
    refs: ['Sun et al. 2024 · Learning to (Learn at Test Time): RNNs with Expressive Hidden States']
  },
  'titans': {
    title: 'Titans：测试时神经记忆（Google 2025）',
    desc: '「惊讶度」驱动记忆：与当前记忆预测不符的信息才写入长期神经记忆模块（MLP 级、可扩展到千万级上下文），配合短期注意力与持久记忆三部分。是 TTT/Memorizing Transformer 一线的工程化代表：记忆在测试时继续学习。',
    latex: [
      'M_t=M_{t-1}-\\theta_t\\,\\nabla\\,\\ell\\big(M_{t-1};x_t\\big)\\quad\\text{(surprise = 梯度大小)}'
    ],
    shapes: ['长期记忆 [d,d] 级','上下文 2M→10M+'],
    code: "const surprise = grad(M => reconLoss(M, x_t));\nM = M.mul(forgetGate).sub(surprise.mul(theta));\ny = attn(x_short) + M.query(q_t);",
    params: '记忆模块 ~MLP',
    refs: ['Behrouz et al. 2025 · Titans: Learning to Memorize at Test Time',
           'Wu et al. 2022 · Memorizing Transformers']
  },
  'looped': {
    title: '深度循环 / 潜空间推理（Universal Transformer → 2025）',
    desc: '同一组层反复迭代、深度按问题难度自适应：简单问题少循环、难题多「想」几轮——把测试时计算从输出 token 转移到潜空间递归。2018 年的 Universal Transformer 在 2025 以 latent reasoning 模型（如 Huginnet）回潮，与 o1 式长思维链互补。',
    latex: [
      'h^{(k+1)}=\\text{Block}\\big(h^{(k)}\\big),\\quad K\\ \\text{按难度自适应（AdaTape/ACT）}'
    ],
    shapes: ['参数量不变','有效深度 K×N'],
    code: "let h = embed(x);\nfor (let k = 0; k < adaptiveSteps(difficulty); k++)\n  h = block(h);   // 权重共享的递归深度",
    params: 'N 层参数（共享复用 K 次）',
    refs: ['Dehghani et al. 2018 · Universal Transformers',
           'Geiping et al. 2025 · Reasoning by Latent Space (Huginnet)']
  },
  'multimodal': {
    title: '多模态融合三路线',
    desc: '①适配器：ViT/Whisper 编码器输出经 projector 接入 LLM（LLaVA、Qwen-VL）——最主流；②早期融合：图文 token 混合统一训练（Chameleon）；③原生全模态：端到端流式输入输出（GPT-4o 系）。共同思想：把一切模态变成 token 序列，交给同一个 Transformer。',
    latex: [
      'h=\\text{LLM}\\big(\\big[\\text{Proj}(v_1..v_m);\\;t_1..t_n\\big]\\big)'
    ],
    shapes: ['图像 → ViT patch tokens → 投影到 d_model','与文本 token 拼接'],
    code: "const vis = proj(vitEncoder(image));       // [m, d]\nconst h = llm(tf.concat([vis, textTokens], 1));",
    params: 'ViT + projector（~LLM 的 5-15%）',
    refs: ['Liu et al. 2023 · Visual Instruction Tuning (LLaVA)',
           'Chameleon Team 2024 · Mixed-Modal Early-Fusion Foundation Models',
           'Bai et al. 2023 · Qwen-VL']
  },

  /* ============ 训练与对齐 ============ */
  'pretrain': {
    title: '预训练（Pre-training）',
    desc: 'LLM 三阶段的第一阶段：在海量无标注文本上做自监督学习——Decoder 系用下一词预测（NTP），BERT 用掩码语言模型（MLM），T5 用 span corruption。语言、知识、基础推理能力全部来自这里；后续对齐阶段只塑造行为，基本不注入新能力（「对齐税」讨论即源于此）。',
    latex: [
      'L_{\\text{NTP}}=-\\sum_t \\log P(y_t\\mid y_{<t}),\\quad L_{\\text{MLM}}=-\\sum_{t\\in M}\\log P(x_t\\mid x_{\\setminus M})'
    ],
    shapes: ['语料 ~10T token → 批次 [B, L]','算力 ~10^25 FLOPs 量级'],
    code: "// 混合 NTP（Decoder）/MLM（Encoder）/span（T5）\nconst loss = tf.losses.softmaxCrossEntropy(oneHot(next), logits);",
    params: '全部模型参数在此阶段学习',
    refs: ['Radford et al. 2018/2019 · GPT 系列', 'Devlin et al. 2018 · BERT',
           'Raffel et al. 2019 · T5']
  },
  'sft': {
    title: 'SFT 指令微调（Supervised Fine-Tuning）',
    desc: '第二阶段：用人类撰写的「指令 → 回答」示范对做监督微调（损失函数与预训练相同，数据变了）。让基座从「会续写」变成「会听指令」。FLAN/T0 证明指令泛化可迁移到未见任务；LIMA 进一步证明对齐强基座只需 ~1000 条高质量示范——数据质量远重于数量。',
    latex: [
      'L_{\\text{SFT}}=-\\sum_t \\log p(y_t\\mid \\text{instruction},\\, y_{<t})'
    ],
    shapes: ['示范对 (指令, 回答) ~1k~1M 条','与预训练同构，仅换数据'],
    code: "// 与预训练同一损失，数据换成指令示范对\nfor (const {prompt, answer} of sftData) loss += ce(model(prompt), answer);",
    params: '全参微调或 LoRA（低秩旁路）',
    refs: ['Wei et al. 2021 · Finetuned Language Models are Zero-Shot Learners (FLAN)',
           'Sanh et al. 2021 · Multitask Prompted Training (T0)',
           'Ouyang et al. 2022 · InstructGPT',
           'Zhou et al. 2023 · LIMA: Less Is More for Alignment']
  },
  'rlhf': {
    title: 'RLHF 人类反馈强化学习',
    desc: '第三阶段（对齐）：①人类对同一提示的多个回答排序 → 训练奖励模型 r(x,y)；②用 PPO 以奖励为信号更新策略，同时用 KL 惩罚锚住 SFT 分布，防止模型钻奖励模型空子（reward hacking）。ChatGPT 的直接来源：让模型从「会补全」变成「有帮助、诚实、无害」的对话者。',
    latex: [
      '\\max_\\theta\\; \\mathbb{E}_{x\\sim D,\\,y\\sim\\pi_\\theta}\\big[r(x,y)\\big] - \\beta\\,\\mathrm{KL}\\big(\\pi_\\theta\\|\\pi_{\\text{SFT}}\\big)'
    ],
    shapes: ['偏好对 (x, y_w, y_l) → 奖励模型','PPO：策略/价值/奖励/参考 四模型'],
    code: "for (const {x, y} of ppoSamples) {\n  const adv = reward(x, y) - beta * kl(policy(x,y), ref(x,y));\n  update(policy, adv);   // PPO 裁剪目标\n}",
    params: '奖励模型 + 策略 + 参考（~3-4 倍推理参数）',
    refs: ['Ziegler et al. 2019 · Fine-Tuning Language Models from Human Preferences',
           'Ouyang et al. 2022 · Training language models to follow instructions (InstructGPT)',
           'Bai et al. 2022 · Training a Helpful and Harmless Assistant (Anthropic)']
  },
  'dpo': {
    title: 'DPO 直接偏好优化（2023）',
    desc: '把 RLHF 的「奖励模型 + PPO」两步折叠为一步：从偏好对出发的闭式变换证明，最优奖励可用策略与参考模型的对数比表示——于是对齐变成一个简单的分类式损失，无需采样、无需显式奖励模型，稳定性与实现难度大幅下降。变体：IPO、KTO、ORPO；工业界 RLHF/DPO 并存。',
    latex: [
      'L_{\\text{DPO}}=-\\log\\,\\sigma\\!\\Big(\\beta\\log\\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)}-\\beta\\log\\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\Big)'
    ],
    shapes: ['输入即偏好对 (x, y_w, y_l)','单模型 + 参考副本'],
    code: "const logRatio = (y) => policy(x, y) - ref(x, y);   // log π/π_ref\nconst loss = -sigmoid(beta * (logRatio(yw) - logRatio(yl)));",
    params: '策略 + 冻结参考（2 倍推理参数，无奖励模型）',
    refs: ['Rafailov et al. 2023 · Direct Preference Optimization',
           'Hong et al. 2024 · ORPO', 'Ethayarajh et al. 2024 · KTO']
  },
  'rlvr': {
    title: '可验证奖励 RL（R1 式推理训练，2024-25）',
    desc: '对齐之后的新浪潮：数学、代码等有客观答案的领域，用「可验证奖励」（答案比对、单元测试通过）做强化学习，模型自发学会长思维链、反思与验证。DeepSeek-R1 证明纯 RL（GRPO：组内相对优势替代 critic）即可涌现推理行为；OpenAI o1 系列为闭源先声——「测试时计算」由此有了训练侧的根基。',
    latex: [
      'R=\\mathbb{1}[\\text{answer correct}],\\quad \\text{GRPO: } A_i=\\frac{r_i-\\text{mean}(r)}{\\text{std}(r)}\\ (\\text{组内相对})'
    ],
    shapes: ['提示 → 长思维链 → 可验证答案',' rollout 长度可达数万 token'],
    code: "for (const x of batch) {\n  const rs = sampleN(policy, x, G).map(y => verify(x, y)); // 0/1\n  const adv = (rs - mean(rs)) / std(rs);                    // GRPO\n  update(policy, adv);\n}",
    params: '策略 + 参考（无 critic）',
    refs: ['DeepSeek-AI 2025 · DeepSeek-R1: Incentivizing Reasoning via RL',
           'Shao et al. 2024 · DeepSeekMath (GRPO)',
           'OpenAI 2024 · Learning to Reason with LLMs (o1)']
  },
  'scaling': {
    title: '缩放定律（Scaling Laws）',
    desc: 'LLM 时代的底层规律：验证损失随参数量 N、数据量 D、算力 C 的幂律平滑下降，可跨数量级外推。Kaplan 2020 给出首组公式；Chinchilla 2022 修正：给定算力，参数与数据应「等比扩展」（70B 配 1.4T token），直接改写了行业训练配方——也解释了 MoE（同等算力换更多参数）与高质量数据竞赛的兴起。',
    latex: [
      'L(N)=L_\\infty+\\Big(\\frac{N_c}{N}\\Big)^{\\alpha},\\quad \\text{Chinchilla: } N_{\\text{opt}},D_{\\text{opt}}\\propto\\sqrt{C}'
    ],
    shapes: ['损失-规模 双对数直线','可提前预测大模型性能'],
    code: "// 用小模型系列拟合幂律，再外推目标规模\nconst {alpha, Nc} = fitPowerLaw(sizes, losses);\nconst lossBig = Linf + Math.pow(Nc / N_big, alpha);",
    params: '拟合参数 α、N_c（非模型参数）',
    refs: ['Kaplan et al. 2020 · Scaling Laws for Neural Language Models',
           'Hoffmann et al. 2022 · Training Compute-Optimal Large Language Models (Chinchilla)']
  },

  /* ============ 前传·基石（1943-2016） ============ */
  'mp': {
    title: 'MP 神经元（1943）',
    desc: 'McCulloch 与 Pitts 把神经元抽象为「加权求和 + 阈值触发」的逻辑单元，并证明任意布尔函数都可由这样的单元组合实现——第一次把「思考」形式化为计算。所有神经网络的零号祖先。',
    latex: ['y=\\mathbb{1}\\Big[\\sum_i w_i x_i \\ge \\theta\\Big]'],
    shapes: ['输入 x_i ∈ {0,1} → 二值输出'],
    code: "const y = (w.reduce((a, wv, i) => a + wv * x[i], 0) >= theta) ? 1 : 0;",
    params: 'n 个权重 + 阈值 θ（手工设定，不可学习）',
    refs: ['McCulloch & Pitts 1943 · A Logical Calculus of the Ideas Immanent in Nervous Activity']
  },
  'hebb': {
    title: 'Hebb 学习规则（1949）',
    desc: '「同时激发的神经元连接在一起」：共同激活的连接应当增强——第一个学习理论，纯局部、无导师。监督学习、无监督学习、对比学习（SimCLR）都是这一思想在不同时代的回声。',
    latex: ['\\Delta w_i = \\eta\\, x_i\\, y'],
    shapes: ['连接强度 w 按共激活增强'],
    code: "w[i] += eta * x[i] * y;   // fire together, wire together",
    params: '学习率 η',
    refs: ['Hebb 1949 · The Organization of Behavior']
  },
  'perceptron': {
    title: '感知机（1958）',
    desc: 'Rosenblatt 给神经元装上「可学习的权重」：误差驱动的更新规则 + 收敛定理（线性可分数据必停），并造出专用硬件 Mark I。第一台真正「学习」的机器——局限在 11 年后由《Perceptrons》揭示。',
    latex: ['w \\leftarrow w + \\eta\\,(t-y)\\,x'],
    shapes: ['线性决策边界','线性可分 ⇒ 必收敛'],
    code: "for (const {x, t} of samples)\n  w = w.add(x.mul(eta * (t - predict(x))));",
    params: 'n 个可学习权重 + 偏置',
    refs: ['Rosenblatt 1958 · The Perceptron: A Probabilistic Model']
  },
  'winter': {
    title: '《Perceptrons》与第一次 AI 寒冬（1969）',
    desc: 'Minsky 与 Papert 数学证明单层感知机连 XOR 都无法表示（线性不可分），叠加算力与数据匮乏、书中对多层路径的悲观论断——神经网络经费枯竭，进入第一次 AI 寒冬。解药（多层 + 非线性 + 反向传播）要到 1986 年才补上。',
    latex: ['XOR \\notin \\text{线性可分} \\Rightarrow \\text{需隐藏层 + 非线性}'],
    shapes: ['单层决策边界：一条直线','XOR 需要两条'],
    code: "// XOR 需要隐藏层：\nh = sigmoid(x.matmul(W1));\ny = sigmoid(h.matmul(W2));",
    params: '—',
    refs: ['Minsky & Papert 1969 · Perceptrons']
  },
  'neocognitron': {
    title: 'Neocognitron（1980）',
    desc: 'Fukushima 模拟视皮层的层次化网络：交替的「特征提取层 + 池化层」逐级抽象边缘 → 形状 → 模式，且对位置平移鲁棒——卷积神经网络（LeNet/AlexNet）的直接思想祖先。',
    latex: ['S 层（特征提取）→ C 层（池化）→ \\cdots 交替'],
    shapes: ['逐层：边缘 → 部件 → 整体'],
    code: "// 层次交替（现代写法）\nconst c1 = maxPool(conv2d(img, K1));\nconst c2 = maxPool(conv2d(c1, K2));",
    params: '各层卷积核',
    refs: ['Fukushima 1980 · Neocognitron: A Self-organizing Neural Network Model']
  },
  'backprop': {
    title: '反向传播（1986）',
    desc: 'Rumelhart、Hinton、Williams 把链式法则系统化用于多层网络：误差从输出层逐层回传、逐层更新权重——多层网络终于「训得动」，深度学习的第一块基石。今天所有 LLM 的训练核心仍是它（配自动微分）。',
    latex: [
      '\\frac{\\partial L}{\\partial w^{(l)}}=\\frac{\\partial L}{\\partial a^{(L)}}\\prod_{k>l}\\frac{\\partial a^{(k)}}{\\partial a^{(k-1)}}\\cdot\\frac{\\partial a^{(l)}}{\\partial w^{(l)}}'
    ],
    shapes: ['前向缓存激活 → 反向逐层回传'],
    code: "// 前向存激活，反向链式乘\nlet delta = lossGrad;\nfor (let l = layers.length - 1; l >= 0; l--) {\n  grads[l] = delta.mul(activations[l].T);\n  delta = delta.matMul(layers[l].W.T).mul(actGrad[l]);\n}",
    params: '—（训练算法）',
    refs: ['Rumelhart, Hinton & Williams 1986 · Learning Representations by Back-propagating Errors']
  },
  'dbn': {
    title: 'DBN 深度信念网络（2006）',
    desc: 'Hinton 用逐层无监督预训练（受限玻尔兹曼机堆叠）+ 有监督微调，让「深层网络能训练」重新可行——深度学习复兴的起点。更深远的是「预训练 + 微调」范式本身：GPT/BERT 的训练哲学是它的直系后代。',
    latex: ['P(v|h)\\;\\text{逐层建模} \\Rightarrow \\text{预训练} \\to \\text{微调}'],
    shapes: ['逐层 RBM 堆叠','预训练 → 微调'],
    code: "// 逐层贪心预训练，再整体微调\nfor (const layer of layers) pretrain(layer, data);\nfineTune(layers, labels);",
    params: '各层 RBM 参数',
    refs: ['Hinton & Salakhutdinov 2006 · A Fast Learning Algorithm for Deep Belief Nets']
  },
  'alexnet': {
    title: 'AlexNet（2012）',
    desc: 'Krizhevsky、Sutskever、Hinton 用 GPU 训练 8 层 CNN（ReLU + Dropout + 数据增强）横扫 ImageNet（top-5 错误率 26%→15%）：「大数据 + GPU 算力 + 深度网络」三驾马车第一次同时到位。深度学习黄金时代引爆点，LLM 的算力路线由此铺开。',
    latex: ['ReLU: \\max(0,x)\\ （比 sigmoid 抗梯度消失）'],
    shapes: ['5 卷积 + 3 全连接','6000 万参数 · 2 块 GTX 580'],
    code: "const y = tf.relu(conv2d(x, K1).maxPool()); // ReLU + 池化\nconst out = fc(y, Wfc);",
    params: '6000 万',
    refs: ['Krizhevsky et al. 2012 · ImageNet Classification with Deep Convolutional Neural Networks']
  },
  'dqn': {
    title: 'DQN 深度强化学习（2013-15）',
    desc: 'DeepMind 用 CNN + 经验回放 + 目标网络，让 AI 直接从屏幕像素学会打 Atari 游戏，达到人类水平——深度强化学习的开端。RL 谱系：DQN（游戏）→ AlphaGo → RLHF（对齐）→ R1（可验证奖励推理），强化学习一路走进 LLM 中心舞台。',
    latex: [
      'Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma\\max_{a\'}Q(s\',a\')-Q(s,a)\\big]'
    ],
    shapes: ['像素 [84,84,4] → Q 值 [|A|]','经验回放池'],
    code: "const target = r + gamma * maxQ(nextState);\nupdate(Q, s, a, target);   // 经验回放采样训练",
    params: 'CNN ~千万',
    refs: ['Mnih et al. 2013/2015 · Human-level Control through Deep Reinforcement Learning']
  },
  'adam': {
    title: 'Adam 优化器（2014）',
    desc: '一阶动矩 + 二阶动矩的偏置校正估计，为每个参数自适应学习率：对超参鲁棒、收敛快、默认即好用。深度学习的事实标准——所有 LLM 至今仍在用其变体 AdamW（解耦权重衰减）。',
    latex: [
      'm_t=\\beta_1 m_{t-1}+(1-\\beta_1)g_t,\\;\\; v_t=\\beta_2 v_{t-1}+(1-\\beta_2)g_t^2',
      '\\theta\\leftarrow\\theta-\\eta\\,\\hat{m}_t/(\\sqrt{\\hat{v}_t}+\\varepsilon)'
    ],
    shapes: ['每个参数 2 个动矩状态'],
    code: "m = b1*m + (1-b1)*g;          // 一阶动矩\nv = b2*v + (1-b2)*g.square(); // 二阶动矩\ntheta -= lr * m.hat().div(v.hat().sqrt().add(eps));",
    params: '每参数 2 倍状态量',
    refs: ['Kingma & Ba 2014 · Adam: A Method for Stochastic Optimization',
           'Loshchilov & Hutter 2017 · AdamW']
  },
  'bn': {
    title: 'Batch Normalization（2015）',
    desc: '对每个 mini-batch 做标准化 + 可学习缩放平移：训练提速、更稳、可用更大学习率。但依赖 batch 统计量，在序列模型/小 batch 下失效——由此演化出按特征维归一化的 LayerNorm（2016），再简化为 RMSNorm（2019）：Transformer 归一化线的历史起点在这里。',
    latex: [
      '\\hat{x}=\\frac{x-\\mu_B}{\\sqrt{\\sigma_B^2+\\varepsilon}}\\cdot\\gamma+\\beta'
    ],
    shapes: ['按 batch 维归一化 [B,L,d]','推理用滑动平均'],
    code: "const mu = x.mean(0, true), var_ = x.variance(0, true);\nconst y = x.sub(mu).div(var_.add(eps).sqrt()).mul(g).add(b);",
    params: 'γ, β + 统计量缓存（2d）',
    refs: ['Ioffe & Szegedy 2015 · Batch Normalization',
           'Ba et al. 2016 · Layer Normalization（演进）']
  },
  'gan': {
    title: 'GAN 生成对抗网络（2014）',
    desc: '生成器 G 与判别器 D 极小极大博弈：G 造假骗过 D，D 升级识破——对抗中 G 学会以假乱真。开启 AI 生成模型时代；与扩散同为生成两大路线，扩散最终在语言上胜出（LLaDA/Mercury），但对抗思想仍活在各类训练技巧中。',
    latex: [
      '\\min_G \\max_D\\; \\mathbb{E}[\\log D(x)]+\\mathbb{E}[\\log(1-D(G(z)))]'
    ],
    shapes: ['噪声 z → G → 假样本 → D → 真/假'],
    code: "const fake = G(z);\nconst dLoss = bce(D(x), 1) + bce(D(fake.detach()), 0);\nconst gLoss = bce(D(fake), 1);   // 骗过 D",
    params: 'G 与 D 各一套',
    refs: ['Goodfellow et al. 2014 · Generative Adversarial Nets']
  },
  'resnet': {
    title: 'ResNet 残差网络（2016）',
    desc: '恒等旁路 y = F(x) + x：让网络只学「残差修正」而非完整映射，梯度有了高速通道——152 层不再是问题，深度退化消失。次年 Transformer 的「Add & Norm」直接继承同一思想：本应用架构图里所有琥珀色虚线旁路都是它。',
    latex: [
      'y=\\mathcal{F}(x,\\{W_i\\})+x\\quad\\text{（恒等映射零成本）}'
    ],
    shapes: ['输入与输出同形 → 可直接相加','深度 152+ 层'],
    code: "const y = block(x).add(x);   // F(x) + x\n// Transformer 里：h = ln(x.add(attn))",
    params: '与普通层相同（旁路零参数）',
    refs: ['He et al. 2016 · Deep Residual Learning for Image Recognition']
  },
  'ebt': {
    title: 'Energy-Based Transformers（2025）',
    desc: '用能量函数 E(x,y) 给「输入-候选答案」打分，推理 = 沿能量下降的迭代优化——把 System 2 式深思做进架构本身，而非靠生成更长的思维链。与 TTT/Titans 同属「测试时计算」路线；训练更稳、泛化更好的主张尚待更大规模验证，2025 年最值得盯的新方向之一。',
    latex: [
      'y^{(k+1)}=y^{(k)}-\\eta\\,\\nabla_y\\,E_\\theta(x,\\,y^{(k)}),\\quad y^{(0)}=\\text{初始猜测}'
    ],
    shapes: ['每步：能量梯度下降','推理算力 ↔ 答案质量 可调'],
    code: "let y = initGuess(x);\nfor (let k = 0; k < K; k++)\n  y = y.sub(gradY(x, y).mul(eta));   // 能量下降\nreturn y;",
    params: '能量网络 E_θ',
    refs: ['Khona et al. 2025 · Energy-Based Transformers are Scalable Learners']
  },
  'v4': {
    title: 'DeepSeek V4：CSA 稀疏注意力 + mHC（2026.4）',
    desc: 'V4-Pro 1.6T 总参/49B 激活（MIT 协议开源），1M 上下文。两大架构更新：①CSA 压缩稀疏注意力 + HCA，把 1M 上下文的 KV Cache 压到 V3.2 的约 10%；②mHC（流形约束超连接）把残差通路加宽并施加流形约束，改进深层信息流。三档推理模式（Non-Think / Think High / Think Max）把「思考深度」做成 API 参数。',
    latex: [
      '\\text{CSA：KV Cache}(1M)\\approx 10\\%\\times V3.2',
      'mHC:\\text{ 残差通路加宽 + 流形约束}'
    ],
    shapes: ['1.6T 总参 / 49B 激活（MoE）','上下文 1M'],
    code: "// API 层切换思考深度\nconst out = v4(prompt, { think: 'high' });",
    params: '1.6T 总参 / 49B 激活',
    refs: ['DeepSeek-AI 2026 · DeepSeek-V4 Technical Report']
  },
  'k3': {
    title: 'Kimi K3：全栈组件替换（2026.7）',
    desc: '首个开源 3T 级模型（2.8T 参数、896 专家选 16、原生视觉、1M 上下文）。Kimi Linear 论文的 2.8T 生产化，全栈组件为推理效率调优：KDA+门控 MLA 混合注意力（69+24 层）、LatentMoE（路由前降维压缩通信）、Attention Residuals（跨层残差按注意力权重加权）、全栈 NoPE——首个彻底去掉 RoPE 的前沿模型，位置信息由因果掩码与 KDA 状态隐式携带。',
    latex: [
      '\\text{K3}=69\\times\\text{KDA}+24\\times\\text{门控MLA}+\\text{LatentMoE}(896\\text{选}16)+\\text{AttnRes}+\\text{NoPE}'
    ],
    shapes: ['2.8T 总参 / ~32B 级激活（1.8%）','原生多模态 · 1M 上下文'],
    code: "// 2026 主线：组件级替换（推理效率调优）\nMoE       → LatentMoE;\nAttention → MLA + KDA 线性混合;\nResidual  → Attention Residuals;\nRoPE      → NoPE;",
    params: '2.8T 总参 / ~32B 激活',
    refs: ['Moonshot AI 2026 · Kimi K3: Open Frontier Intelligence',
           'Raschka 2026 · Kimi K3 Architecture Notes']
  }
};

const ARCH_NODES = [
  /* ---- Encoder 列 ---- */
  { id:'input',    x:80,  y:48,  label:'输入 Tokens',            sub:'分词后的索引序列', detail:'input',
    tip:'源句分词得到的 token 索引序列' },
  { id:'embed',    x:80,  y:112, label:'Embedding',              sub:'查表后 × √d_model', detail:'embed',
    tip:'词向量查表并放大，使与位置编码量级匹配' },
  { id:'pe',       x:80,  y:176, label:'+ Positional Encoding',  detail:'pe',
    tip:'正弦位置编码，为模型注入词序信息' },
  { id:'enc-mha',  x:80,  y:262, label:'多头自注意力',           sub:'Self-Attention（全并行）', detail:'enc-mha',
    tip:'每个位置关注同句所有位置，一次矩阵运算完成' },
  { id:'enc-an1',  x:80,  y:320, label:'Add & Norm',             detail:'addnorm',
    tip:'残差连接稳定梯度，层归一化稳定分布' },
  { id:'enc-ffn',  x:80,  y:378, label:'前馈网络 FFN',           sub:'d_model → d_ff → d_model', detail:'ffn',
    tip:'逐位置两层 MLP，参数量的主要来源' },
  { id:'enc-an2',  x:80,  y:436, label:'Add & Norm',             detail:'addnorm',
    tip:'第二个残差 + 归一化子层' },
  /* ---- Decoder 列 ---- */
  { id:'dec-input',      x:458, y:48,  label:'目标输入（右移一位）', sub:'以 <bos> 开头', detail:'input',
    tip:'训练时整体右移；推理时逐步喂入已生成前缀' },
  { id:'dec-embed',      x:458, y:112, label:'Embedding + 位置编码', detail:'embed',
    tip:'目标序列的嵌入与位置编码' },
  { id:'dec-masked-mha', x:458, y:198, label:'掩码多头自注意力',     sub:'只允许看 j ≤ i 的位置', detail:'dec-masked-mha',
    tip:'因果掩码防止偷看未来位置' },
  { id:'dec-an1',        x:458, y:256, label:'Add & Norm', detail:'addnorm', tip:'残差 + 层归一化' },
  { id:'dec-cross',      x:458, y:314, label:'交叉注意力', sub:'Q←解码器，K/V←编码器', detail:'dec-cross',
    tip:'解码器读取源句信息的唯一入口' },
  { id:'dec-an2',        x:458, y:372, label:'Add & Norm', detail:'addnorm', tip:'残差 + 层归一化' },
  { id:'dec-ffn',        x:458, y:430, label:'前馈网络 FFN', detail:'ffn', tip:'逐位置两层 MLP' },
  { id:'dec-an3',        x:458, y:488, label:'Add & Norm', detail:'addnorm', tip:'残差 + 层归一化' },
  { id:'out',            x:458, y:568, label:'Linear + Softmax', sub:'投影到词表 |V|', detail:'out',
    tip:'末位置隐状态 → 全词表概率分布' },
  { id:'outp',           x:458, y:632, label:'输出概率 P(u|x)', sub:'argmax 作为下一步输入', detail:'out',
    tip:'取概率最大者作为下一个 token，循环生成' }
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
  { d:'M 276 457 C 350 457, 384 335, 458 335', label:'K, V（编码器记忆）', lx:352, ly:378 },
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
    { x:330, y:92,  text:'① 循环时代：状态逐步传递，长程依赖路径 O(t)' },
    { x:556, y:312, text:'② 注意力革命：任意位置直连，路径 O(1)' }
  ],
  nodes: [
    { id:'rnn', x:24, y:120, label:'简单 RNN (1990)', sub:'hₜ = φ(W·hₜ₋₁ + U·xₜ)', detail:'rnn',
      tip:'一切序列建模的起点：隐藏状态沿时间递推' },
    { id:'lstm', x:282, y:120, label:'LSTM (1997) / GRU (2014)', sub:'门控记忆，缓解梯度消失', detail:'lstm',
      tip:'输入/遗忘/输出门控细胞状态；GRU 是其简化版' },
    { id:'s2s', x:540, y:120, label:'Seq2Seq (2014)', sub:'Encoder-Decoder + 固定向量瓶颈', detail:'s2s',
      tip:'机器翻译范式确立，但整句压进一个定长向量' },
    { id:'batt', x:282, y:340, label:'加性注意力 (2015)', sub:'解码器逐步「回看」全部编码状态', detail:'batt',
      tip:'Bahdanau 注意力：对齐权重 αₜᵢ 动态加权编码状态（同年 Luong 提出乘性版本）' },
    { id:'selfattn', x:540, y:340, label:'自注意力 (2017)', sub:'Q·K·V 全并行 → 见「2017 原版」', detail:'selfattn',
      tip:'注意力从「解码器看编码器」推广为「序列看自身」，取代循环' }
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
    { id:'tok', x:282, y:36, label:'输入 Tokens', sub:'分词 → 词表索引', detail:'input',
      tip:'BPE 分词后的 token 索引序列' },
    { id:'emb', x:282, y:92, label:'Embedding', sub:'查表后 × √d_model', detail:'embed',
      tip:'词向量查表并放大（LLaMA 起不再乘 √d）' },
    { id:'rope', x:282, y:148, label:'+ RoPE', sub:'旋转位置编码（绝对位置·相对表达）', detail:'rope',
      tip:'把位置编码为二维旋转，内积只依赖相对位置；ALiBi/YaRN 等为变体' },
    { id:'rms1', x:282, y:218, label:'RMSNorm', sub:'Pre-Norm（先归一化再子层）', detail:'rmsnorm',
      tip:'Pre-LN/RMSNorm 让深层网络训练稳定，取代 2017 的 Post-LN' },
    { id:'gqa', x:282, y:276, label:'GQA 掩码自注意力', sub:'因果掩码 · KV 头分组共享', detail:'gqa',
      tip:'Qwen/LLaMA-2+ 的标配：KV 头少于 Q 头，Cache 直接缩小' },
    { id:'res1', x:282, y:334, label:'⊕ 残差', detail:'addnorm', tip:'残差直连，梯度高速通道' },
    { id:'rms2', x:282, y:392, label:'RMSNorm', detail:'rmsnorm', tip:'第二个 Pre-Norm' },
    { id:'swiglu', x:282, y:450, label:'SwiGLU FFN', sub:'门控：d → d_ff → d', detail:'swiglu',
      tip:'SiLU 门控分支 ⊙ 线性分支，取代 ReLU FFN' },
    { id:'res2', x:282, y:508, label:'⊕ 残差', detail:'addnorm', tip:'第二处残差' },
    { id:'lnf', x:282, y:580, label:'Final RMSNorm', detail:'rmsnorm', tip:'输出前的最后一次归一化' },
    { id:'head', x:282, y:638, label:'LM Head · Softmax(|V|)', sub:'常与 Embedding 权重共享', detail:'lm-head',
      tip:'隐状态 → 全词表分布，逐 token 自回归' }
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
    { id:'tok', x:232, y:36, label:'输入 Tokens', sub:'分词 → 词表索引', detail:'input', tip:'同 Dense 视图' },
    { id:'emb', x:232, y:92, label:'Embedding', detail:'embed', tip:'词向量查表' },
    { id:'rope', x:232, y:148, label:'+ RoPE', detail:'rope', tip:'旋转位置编码' },
    { id:'rms1', x:232, y:218, label:'RMSNorm', sub:'Pre-Norm', detail:'rmsnorm', tip:'Pre-Norm 归一化' },
    { id:'gqa', x:232, y:276, label:'GQA 掩码自注意力', detail:'gqa', tip:'KV 分组共享的因果自注意力' },
    { id:'an1', x:232, y:334, label:'Add & Norm', detail:'addnorm', tip:'注意力残差 + 归一化' },
    { id:'rms2', x:232, y:392, label:'RMSNorm', detail:'rmsnorm', tip:'MoE 子层的 Pre-Norm' },
    { id:'router', x:232, y:450, label:'MoE Router 门控', sub:'softmax → Top-K 稀疏激活', detail:'moe-router',
      tip:'每个 token 只选 K 个专家：参数规模与计算量解耦' },
    { id:'wsum', x:232, y:508, label:'稀疏加权合成', sub:'y = Σ gᵢ · Expertᵢ(x)', detail:'moe-sum',
      tip:'仅被选中的专家输出参与合成' },
    { id:'an2', x:232, y:566, label:'Add & Norm', detail:'addnorm', tip:'MoE 残差 + 归一化' },
    { id:'lnf', x:232, y:624, label:'Final RMSNorm + LM Head', sub:'Linear → Softmax(|V|)', detail:'lm-head',
      tip:'输出层（DeepSeek-V3 的 MTP 头略）' },
    { id:'exp1', x:532, y:406, label:'专家 1 · SwiGLU', detail:'moe-expert', tip:'一个专家 = 一个 SwiGLU FFN' },
    { id:'exp2', x:532, y:454, label:'专家 2 · SwiGLU', detail:'moe-expert', tip:'一个专家 = 一个 SwiGLU FFN' },
    { id:'exp3', x:532, y:502, label:'专家 3 · SwiGLU', detail:'moe-expert', tip:'一个专家 = 一个 SwiGLU FFN' },
    { id:'exp4', x:532, y:550, label:'专家 E · SwiGLU', sub:'（示意 4 个，实际 64~384 个）', detail:'moe-expert',
      tip:'Qwen3: 128 选 8 · K2: 384 选 8 · DeepSeek-V3: 256 选 8 + 1 共享' }
  ],
  frames: [
    { x:214, y:196, w:232, h:394, label:'Decoder Block × N' },
    { x:514, y:388, w:232, h:204, label:'Experts × E · Top-K 激活' }
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
    { x:370, y:90,  text:'路线 A · KV Cache 压缩（推理显存/带宽）' },
    { x:370, y:300, text:'路线 B · 计算与访存效率（长上下文）' },
    { x:370, y:510, text:'路线 C · 上下文扩展与训练稳定化' }
  ],
  nodes: [
    { id:'mha-ev', x:20, y:120, label:'MHA 多头注意力', sub:'2017 · h 组独立 KV', detail:'mha-ev',
      tip:'基线：KV Cache = 2·L·h·dₕ' },
    { id:'mqa', x:200, y:120, label:'MQA 多查询', sub:'2019 · 全头共享 1 组 KV', detail:'mqa',
      tip:'Shazeer：One Write-Head is All You Need' },
    { id:'gqa', x:380, y:120, label:'GQA 分组查询', sub:'2023 · 分组共享（Qwen 系）', detail:'gqa',
      tip:'MQA 与 MHA 的折中，Uptraining 转换；LLaMA-2/3、Qwen 标配' },
    { id:'mla', x:560, y:120, label:'MLA 潜在注意力', sub:'2024 · 低秩压缩（DeepSeek）', detail:'mla',
      tip:'Cache 只存潜在向量 c，按需升维还原 K/V；K2 沿用' },
    { id:'flash', x:20, y:330, label:'FlashAttention', sub:'2022 · 分块在线 softmax', detail:'flash',
      tip:'IO 感知内核：数学不变，显存 O(L²)→O(L)；v2/v3 持续进化' },
    { id:'sparse', x:200, y:330, label:'稀疏注意力', sub:'2025 · NSA / MoBA', detail:'sparse',
      tip:'只算重要块且稀疏模式可训练：DeepSeek NSA、Kimi MoBA' },
    { id:'linear', x:380, y:330, label:'线性注意力', sub:'2020→25 · 核化 O(Ld²)', detail:'linear',
      tip:'φ(Q)(φ(K)ᵀV)：与序列长度线性，见「线性·SSM」视图' },
    { id:'hybrid', x:560, y:330, label:'混合架构', sub:'Jamba · Griffin · KDA', detail:'hybrid',
      tip:'少数全注意力层 + 多数线性层按比例混搭（Kimi Linear 等）' },
    { id:'swa', x:20, y:540, label:'滑动窗口 SWA', sub:'2023 · 局部/全局 5:1 交错', detail:'swa',
      tip:'Mistral/Gemma：每层只看窗口 w，全局层稀疏布点' },
    { id:'ctxext', x:200, y:540, label:'上下文扩展', sub:'PI · NTK · YaRN', detail:'ctx-ext',
      tip:'RoPE 训练长度→百万级推理：按频率插值外推' },
    { id:'qknorm', x:380, y:540, label:'训练稳定化', sub:'QK-Norm · soft-cap · z-loss', detail:'qknorm',
      tip:'大模型训练三大稳定器（Gemma 2/3、Chameleon）' },
    { id:'diff', x:560, y:540, label:'差分注意力', sub:'2024 · 双 softmax 消噪', detail:'diff',
      tip:'DIFF Transformer：差分放大器思路，抗幻觉' }
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
    { x:178, y:60, text:'支线 1 · 核化线性注意力' },
    { x:556, y:60, text:'支线 2 · 结构化状态空间（SSM）' }
  ],
  nodes: [
    { id:'lin', x:80, y:100, label:'线性 Transformer (2020)', sub:'Transformers are RNNs', detail:'linear',
      tip:'φ 核化 + 结合律，摆脱 L²' },
    { id:'rwkv', x:80, y:210, label:'RWKV (2022)', sub:'通道级衰减 · 可并行可递推', detail:'rwkv',
      tip:'把注意力改写为带衰减的 RNN' },
    { id:'xlstm', x:80, y:320, label:'xLSTM (2024)', sub:'指数记忆 + 矩阵状态', detail:'xlstm',
      tip:'LSTM 原作者的复兴之作' },
    { id:'s4', x:458, y:100, label:'S4 (2021)', sub:'HiPPO + 结构化 SSM', detail:'s4',
      tip:'连续状态空间 + 核化卷积' },
    { id:'mamba', x:458, y:210, label:'Mamba (2023)', sub:'选择机制 · 硬件感知扫描', detail:'mamba',
      tip:'让 SSM 参数依赖输入 → 可内容筛选；Mamba-2 (2024) 与注意力统一' },
    { id:'kda', x:269, y:440, label:'Kimi Linear · KDA (2025)', sub:'门控增量记忆 3/4 层 + MLA 1/4 层', detail:'kda',
      tip:'Kimi 的混合线性架构：百万级上下文、显存与吞吐双优' }
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
    { x:380, y:26, text:'生成范式：自回归及其挑战者' }
  ],
  nodes: [
    { id:'autoreg', x:282, y:48, label:'自回归 NTP（主流）', sub:'GPT 系 · 逐 token 左到右', detail:'autoreg',
      tip:'P(y)=∏P(yₜ|y<ₜ)：所有主流 LLM 的训练与生成范式' },
    { id:'mtp', x:60, y:210, label:'MTP 多 token 预测', sub:'DeepSeek-V3 · 一次预测多步', detail:'mtp',
      tip:'辅助头密集训练信号；推理时可当投机草稿' },
    { id:'spec', x:504, y:210, label:'投机解码', sub:'草稿模型 + 并行验证（EAGLE）', detail:'spec',
      tip:'数学无损的 2-4× 解码加速；属推理系统层，非架构',
      dashed:true, badge:'推理层' },
    { id:'diffusion', x:282, y:380, label:'扩散语言模型', sub:'LLaDA · Mercury 并行去噪', detail:'diffusion',
      tip:'从全掩码迭代去噪：打破左到右依赖' }
  ],
  frames: [],
  edges: [],
  special: [
    { d:'M 300 90 C 220 130, 180 170, 158 208' },              /* autoreg → mtp */
    { d:'M 460 90 C 540 130, 580 170, 588 208' },              /* autoreg → spec */
    { d:'M 256 231 L 504 231', label:'草稿 / 验证', lx:380, ly:224 },
    { d:'M 380 90 L 380 378',  label:'范式挑战 · 并行去噪', lx:462, ly:300 }
  ],
  residuals: []
};

const ARCH_VIEW_FRONTIER = {
  colTitles: [
    { x:178, y:60, text:'去分词器与记忆层' },
    { x:556, y:60, text:'测试时计算与深度' },
    { x:380, y:420, text:'模态扩展：一切皆 Token' }
  ],
  nodes: [
    { id:'blt', x:80, y:100, label:'BLT 字节级 LM', sub:'2024 · 熵驱动动态 patch', detail:'blt',
      tip:'Meta：去 BPE，计算量随信息熵分配' },
    { id:'memlayer', x:80, y:210, label:'记忆层 Memory', sub:'2024 · 稀疏键值替代 FFN', detail:'memlayer',
      tip:'知识存记忆、推理靠注意力；事实性大幅提升' },
    { id:'looped', x:458, y:100, label:'深度循环 / 潜推理', sub:'2018→25 · 递归深度自适应', detail:'looped',
      tip:'Universal Transformer 回潮：难题多转几圈' },
    { id:'ttt', x:458, y:210, label:'TTT 层', sub:'2024 · 隐藏状态=在线学习器', detail:'ttt',
      tip:'序列建模即测试时训练' },
    { id:'titans', x:269, y:330, label:'Titans 神经记忆', sub:'2025 · 惊讶度驱动 · 千万级上下文', detail:'titans',
      tip:'Google：长期记忆与短期注意力分工' },
    { id:'multimodal', x:269, y:440, label:'多模态融合', sub:'LLaVA · Chameleon · 原生 Omni', detail:'multimodal',
      tip:'视觉/音频统一成 token 进 Transformer' },
    { id:'ebt', x:470, y:330, label:'EBT 能量 Transformer', sub:'2025 · 能量函数隐式推理', detail:'ebt',
      tip:'推理 = 能量最小化的迭代下降；测试时计算新分支' }
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
    { x:330, y:60,  text:'三阶段范式：预训练 → 指令微调 → 对齐' },
    { x:330, y:200, text:'关键分支与底层规律' }
  ],
  nodes: [
    { id:'pretrain', x:24, y:100, label:'预训练 Pre-training',
      sub:'自监督 NTP/MLM · 万亿 token', detail:'pretrain',
      tip:'语言/知识/推理底子的来源；对齐阶段不注入新能力' },
    { id:'sft', x:282, y:100, label:'SFT 指令微调',
      sub:'2021/22 · 人类示范监督学习', detail:'sft',
      tip:'让基座「听懂指令」；数据质量 >> 数量（LIMA）' },
    { id:'rlhf', x:540, y:100, label:'RLHF 强化学习',
      sub:'2022 · 奖励模型 + PPO 对齐', detail:'rlhf',
      tip:'从「会补全」到「会对话」——ChatGPT 的直接来源' },
    { id:'scaling', x:24, y:240, label:'缩放定律',
      sub:'2020/22 · Kaplan / Chinchilla', detail:'scaling',
      tip:'损失随规模幂律下降；参数与数据等比扩展' },
    { id:'rlvr', x:282, y:240, label:'可验证奖励 RL',
      sub:'2024-25 · R1 式长推理链训练', detail:'rlvr',
      tip:'数学/代码用标准答案做奖励；GRPO 免 critic' },
    { id:'dpo', x:540, y:240, label:'DPO 直接偏好优化',
      sub:'2023 · 免显式奖励模型', detail:'dpo',
      tip:'把 RLHF 两步折叠为一步分类式损失' }
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
    { x:370, y:60,  text:'理论奠基 1943-1969' },
    { x:370, y:230, text:'复兴与突破 1980-2012' },
    { x:370, y:400, text:'现代基石 2014-2016' }
  ],
  nodes: [
    { id:'mp', x:24, y:110, label:'MP 神经元 (1943)',
      sub:'加权求和 + 阈值触发', detail:'mp',
      tip:'神经网络的数学原点：任意布尔函数可由其组合实现' },
    { id:'hebb', x:208, y:110, label:'Hebb 规则 (1949)',
      sub:'共同激活 → 连接增强', detail:'hebb',
      tip:'第一个学习理论；无监督/对比学习的思想源头' },
    { id:'perceptron', x:392, y:110, label:'感知机 (1958)',
      sub:'可学习的线性分类器', detail:'perceptron',
      tip:'第一台真正「学习」的机器（当时是硬件）' },
    { id:'winter', x:576, y:110, label:'《Perceptrons》(1969)',
      sub:'XOR 不可分 → AI 寒冬', detail:'winter',
      tip:'单层不够：需要多层 + 非线性（1986 补上）' },
    { id:'neocognitron', x:24, y:280, label:'Neocognitron (1980)',
      sub:'层次特征提取 → CNN 前身', detail:'neocognitron',
      tip:'模拟视皮层的交替结构：卷积+池化' },
    { id:'backprop', x:208, y:280, label:'反向传播 (1986)',
      sub:'链式法则逐层回传', detail:'backprop',
      tip:'多层网络终于「训得动」——今天所有 LLM 的训练核心' },
    { id:'dbn', x:392, y:280, label:'DBN (2006)',
      sub:'逐层预训练 + 微调', detail:'dbn',
      tip:'深度学习复兴起点；预训练+微调范式的第一次胜利' },
    { id:'alexnet', x:576, y:280, label:'AlexNet (2012)',
      sub:'GPU + 大数据 引爆', detail:'alexnet',
      tip:'三驾马车第一次同时到位；LLM 算力路线由此铺开' },
    { id:'adam', x:24, y:450, label:'Adam (2014)',
      sub:'自适应动矩优化器', detail:'adam',
      tip:'所有 LLM 至今的默认优化器（AdamW 变体）' },
    { id:'gan', x:208, y:450, label:'GAN (2014)',
      sub:'生成对抗 · 生成模型起点', detail:'gan',
      tip:'对抗/生成思想 → 扩散模型（LLaDA/Mercury）' },
    { id:'bn', x:392, y:450, label:'BatchNorm (2015)',
      sub:'→ LayerNorm → RMSNorm', detail:'bn',
      tip:'归一化演进线的起点；序列模型最终选择 RMSNorm' },
    { id:'resnet', x:576, y:450, label:'ResNet (2016)',
      sub:'残差旁路 → Transformer Add', detail:'resnet',
      tip:'架构图里所有琥珀虚线旁路都是它' }
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
  { from:1943, to:1989, label:'第一阶段 · 神经网络诞生（1943-1986）',
    meaning:'计算机拥有神经元，找到多层网络训练方法' },
  { from:1990, to:2016, label:'第二阶段 · 深度学习复兴与突破（1990-2016）',
    meaning:'数据、算法、算力三驾马车，深度学习进入黄金时代' },
  { from:2017, to:2022, label:'第三阶段 · Transformer 时代（2017-2022）',
    meaning:'模型规模爆发，Transformer 成为通用基础架构' },
  { from:2023, to:2100, label:'第四阶段 · 后 Transformer 探索（2023-2026）',
    meaning:'追求高效率、长上下文、低成本，迈向通用智能' }
];

const HISTORY = [
  /* ---- 第一阶段：神经网络诞生 1943-1986 ---- */
  { year:1943, view:'dawn', name:'MP 神经元',
    paper:'A Logical Calculus of the Ideas Immanent in Nervous Activity (McCulloch & Pitts)',
    contrib:'神经活动逻辑演算，人工神经网络理论基石',
    highlights:['mp'],
    note:'一切的开始：把「思考」形式化为加权求和 + 阈值触发。' },
  { year:1949, view:'dawn', name:'Hebb 学习规则',
    paper:'The Organization of Behavior (Hebb)',
    contrib:'共同激活强化神经元连接：第一个学习规则',
    highlights:['hebb','mp'],
    note:'「fire together, wire together」——无监督与对比学习思想的源头。' },
  { year:1958, view:'dawn', name:'感知机',
    paper:'The Perceptron (Rosenblatt)',
    contrib:'可学习权重 + 收敛定理，开启神经网络研究先河',
    highlights:['perceptron'],
    note:'第一台真正「学习」的机器（硬件 Mark I），媒体一度预言机器将「走路、说话、有意识」。' },
  { year:1969, view:'dawn', name:'《Perceptrons》与寒冬',
    paper:'Perceptrons (Minsky & Papert)',
    contrib:'证明单层无法表示 XOR，推动理论反思',
    highlights:['winter'],
    note:'一本书冻结一个领域：经费枯竭，第一次 AI 寒冬——解药（多层+反向传播）17 年后才到。' },
  { year:1980, view:'dawn', name:'Neocognitron',
    paper:'Neocognitron: A Self-organizing Neural Network Model (Fukushima)',
    contrib:'层次化神经网络，卷积思想基础',
    highlights:['neocognitron'],
    note:'交替的特征提取与池化层——十年后 LeNet、二十年后 AlexNet 都在重复这个结构。' },
  { year:1986, view:'dawn', name:'反向传播',
    paper:'Learning Representations by Back-propagating Errors (Rumelhart, Hinton & Williams)',
    contrib:'链式法则逐层回传，多层网络有效训练',
    highlights:['backprop'],
    note:'寒冬的解药终于到来：多层网络第一次训得动。今天每个 LLM 训练的核心仍是它。' },
  /* ---- 前史：循环时代 ---- */
  { year:1990, view:'pre', name:'简单 RNN',
    paper:'Finding Structure in Time (Elman)',
    contrib:'一切序列建模的起点：隐藏状态沿时间递推',
    highlights:['rnn'],
    note:'最原始的架构：t 时刻必须等 t−1 算完，梯度连乘导致只能记住约 10~20 步——并行与长程依赖是贯穿 30 年的两道坎。' },
  { year:1997, view:'pre', name:'LSTM',
    paper:'Long Short-Term Memory (Hochreiter & Schmidhuber)',
    contrib:'门控记忆单元，缓解循环网络的梯度消失',
    highlights:['rnn','lstm'],
    note:'给 RNN 装上传送带：细胞状态近似线性流动。统治序列建模 15 年（GRU 是其 2014 年简化版）。' },
  { year:1998, view:'dawn', name:'LeNet（CNN 落地）',
    paper:'Gradient-Based Learning Applied to Document Recognition (LeCun et al.)',
    contrib:'卷积网络落地手写数字识别',
    highlights:['neocognitron'],
    note:'CNN 第一次工业级应用（美国支票识别）——「端到端学习取代特征工程」的早期示范。' },
  { year:2006, view:'dawn', name:'DBN 深度信念网络',
    paper:'A Fast Learning Algorithm for Deep Belief Nets (Hinton & Salakhutdinov)',
    contrib:'逐层预训练：深度网络高效训练算法',
    highlights:['dbn','backprop'],
    note:'深度学习复兴的起点；「预训练 + 微调」范式第一次胜利——GPT/BERT 是它的精神后代。' },
  { year:2012, view:'dawn', name:'AlexNet',
    paper:'ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al.)',
    contrib:'GPU+ReLU+Dropout 夺冠，引爆深度学习',
    highlights:['alexnet','backprop'],
    note:'三驾马车（数据/算法/算力）第一次同时到位——此后一切大模型故事的地基。' },
  { year:2013, view:'train', name:'DQN 深度强化学习',
    paper:'Playing Atari with Deep Reinforcement Learning (Mnih et al.)',
    contrib:'像素级自学游戏：深度 RL 开山',
    highlights:['rlvr'],
    note:'RL 谱系：DQN → AlphaGo → RLHF → R1——强化学习一路走进 LLM 中心舞台。' },
  { year:2014, view:'pre', name:'Seq2Seq + GRU',
    paper:'Sequence to Sequence Learning (Sutskever et al.) · Learning Phrase Representations (Cho et al.)',
    contrib:'编码器-解码器框架确立，机器翻译神经网络化',
    highlights:['lstm','s2s'],
    note:'变长→变长的通用范式诞生；但整句被压进一个固定向量，长句信息溢出——瓶颈即机会。' },
  { year:2014, view:'dawn', name:'GAN 生成对抗',
    paper:'Generative Adversarial Nets (Goodfellow et al.)',
    contrib:'开启 AI 生成模型新时代',
    highlights:['gan'],
    note:'生成器与判别器博弈式训练——生成范式线（扩散 LM）的思想远祖。' },
  { year:2014, view:'dawn', name:'Adam 优化器',
    paper:'Adam: A Method for Stochastic Optimization (Kingma & Ba)',
    contrib:'自适应动矩优化，训练效率大增',
    highlights:['adam'],
    note:'默认即好用的优化器：从感知机到 GPT，深度学习的事实标准（LLM 用 AdamW 变体）。' },
  { year:2015, view:'pre', name:'注意力机制',
    paper:'Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al.)',
    contrib:'解码器逐步「回看」全部编码状态，软对齐',
    highlights:['s2s','batt','selfattn'],
    note:'注意力的起源：α 权重动态加权编码状态（同年 Luong 提出点积版本）。只差一步——把注意力推广到序列自身。' },
  /* ---- Transformer 与三大范式 ---- */
  { year:2015, view:'attn', name:'BatchNorm',
    paper:'Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy)',
    contrib:'归一化加速训练，提升稳定性',
    highlights:['qknorm'],
    note:'归一化演进线起点：BN（按 batch）→ LayerNorm（按特征）→ RMSNorm——Transformer 最终选择后者，BN 对序列/小 batch 不友好是主因。' },
  { year:2016, view:'dense', name:'ResNet 残差网络',
    paper:'Deep Residual Learning for Image Recognition (He et al.)',
    contrib:'残差连接，解决深度网络退化问题',
    highlights:['res1','res2'],
    note:'y = F(x) + x：梯度高速通道。次年 Transformer 的「Add & Norm」直接继承——切到 Dense 视图看到的琥珀虚线就是它。' },
  { year:2017, view:'orig', name:'Transformer', primary:true,
    paper:'Attention Is All You Need (Vaswani et al.)',
    contrib:'完全基于注意力，抛弃循环与卷积，训练全并行',
    highlights:['embed','pe','enc-mha','enc-an1','enc-ffn','enc-an2','dec-masked-mha','dec-cross'],
    note:'分水岭：自注意力让任意两位置 O(1) 直连。此后裂变为三大范式——只用 Encoder（BERT）、只用 Decoder（GPT）、完整 Enc-Dec（T5）。' },
  { year:2018, view:'orig', name:'GPT-1（Decoder-Only 开端）',
    paper:'Improving Language Understanding by Generative Pre-Training (Radford et al.)',
    contrib:'只用 Decoder + 生成式预训练 + 微调',
    highlights:['dec-input','dec-embed','dec-masked-mha','dec-an1','out','outp'],
    note:'「Decoder-Only 也能做理解任务」：12 层 Block + LM Head，自回归目标一用到底——GPT 系路线图的起点。' },
  { year:2018, view:'orig', name:'BERT（Encoder-Only）',
    paper:'BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.)',
    contrib:'双向编码器预训练 + 微调范式，横扫 NLU 榜单',
    highlights:['embed','pe','enc-mha','enc-an1','enc-ffn','enc-an2'],
    note:'只用 Encoder：双向上下文适合理解类任务（RoBERTa 2019 进一步打磨训练配方）。' },
  { year:2019, view:'orig', name:'GPT-2',
    paper:'Language Models are Unsupervised Multitask Learners (Radford et al.)',
    contrib:'15 亿参数解码器语言模型，零样本多任务；Pre-LN 确立',
    highlights:['dec-input','dec-embed','dec-masked-mha','dec-an1','dec-ffn','dec-an3','out','outp'],
    note:'只用 Decoder：因果掩码自回归生成，规模即能力；Pre-LN 取代 Post-LN，深层训练从此稳定。' },
  { year:2019, view:'orig', name:'T5（Enc-Dec 复兴）',
    paper:'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al.)',
    contrib:'一切任务皆文本到文本，Encoder-Decoder 规模化验证',
    highlights:['input','embed','enc-mha','dec-cross','out'],
    note:'Encoder-Decoder 路线的集大成者（11B）；翻译/摘要等有「输入-输出」结构的任务至今仍常用此范式。' },
  { year:2019, view:'train', name:'RLHF 起源',
    paper:'Fine-Tuning Language Models from Human Preferences (Ziegler et al.)',
    contrib:'首次系统化：人类偏好 → 奖励 → 强化学习',
    highlights:['rlhf'],
    note:'OpenAI 与 Anthropic 共同的起点：让模型对齐人类偏好，比「训得更大」早了三年。' },
  { year:2019, view:'attn', name:'MQA 多查询注意力',
    paper:'Fast Transformer Decoding: One Write-Head is All You Need (Shazeer)',
    contrib:'全头共享一组 KV，推理 Cache 骤降 h 倍',
    highlights:['mha-ev','mqa'],
    note:'效率演进线的第一枪：发现推理瓶颈在 KV Cache 而非参数——此后 MQA→GQA→MLA 一脉相承。' },
  { year:2020, view:'orig', name:'GPT-3',
    paper:'Language Models are Few-Shot Learners (Brown et al.)',
    contrib:'1750 亿参数，上下文少样本学习（few-shot）',
    highlights:['dec-input','dec-masked-mha','out','outp'],
    note:'结构与 GPT-2 相同，靠规模涌现出上下文学习——大模型时代的起点（ChatGPT 即其 RLHF 后代）。' },
  { year:2020, view:'train', name:'缩放定律',
    paper:'Scaling Laws for Neural Language Models (Kaplan et al.)',
    contrib:'损失随参数/数据/算力幂律下降，可跨数量级外推',
    highlights:['scaling'],
    note:'「大力出奇迹」的定量依据：训练前就能预测大模型性能——2022 年 Chinchilla 修正为参数与数据等比扩展。' },
  { year:2020, view:'orig', name:'ViT（跨界）',
    paper:'An Image is Worth 16x16 Words (Dosovitskiy et al.)',
    contrib:'图像切 patch 当 token：Transformer 统一视觉',
    highlights:['input','embed','enc-mha','enc-an1'],
    note:'证明自注意力与模态无关——此后语音(Whisper)、蛋白质(AlphaFold2)全面 Transformer 化。' },
  { year:2020, view:'ssm', name:'线性 Transformer',
    paper:'Transformers are RNNs (Katharopoulos et al.)',
    contrib:'φ 核化 + 结合律，复杂度 O(L²)→O(Ld²)',
    highlights:['lin'],
    note:'线性注意力开山：「Transformer 就是 RNN」。效率线与循环复兴线在此交汇。' },
  { year:2021, view:'moe', name:'Switch Transformer',
    paper:'Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al.)',
    contrib:'Top-1 稀疏路由，万亿参数 MoE（GShard 2020 首创 MoE+Transformer）',
    highlights:['router','exp1','exp2','exp3','exp4','wsum'],
    note:'MoE 与 Transformer 合体：总参数与单 token 计算解耦——万亿时代的钥匙。' },
  { year:2021, view:'ssm', name:'S4',
    paper:'Efficiently Modeling Long Sequences with Structured State Spaces (Gu et al.)',
    contrib:'HiPPO + 结构化 SSM，长序列基准横扫',
    highlights:['s4'],
    note:'状态空间模型回归：连续系统离散化，训练当卷积、推理当 RNN——Mamba 的直系前身。' },
  { year:2021, view:'train', name:'FLAN 指令微调',
    paper:'Finetuned Language Models are Zero-Shot Learners (Wei et al.)',
    contrib:'指令化任务数据 → 未见任务零样本泛化',
    highlights:['sft','pretrain'],
    note:'「指令微调有效」的奠基工作：SFT 阶段由此成为标配，直接催生 InstructGPT 三阶段范式。' },
  { year:2022, view:'train', name:'InstructGPT / ChatGPT',
    paper:'Training language models to follow instructions with human feedback (Ouyang et al.)',
    contrib:'SFT → RLHF → PPO 三阶段范式确立',
    highlights:['sft','rlhf','pretrain'],
    note:'1.3B 的 InstructGPT 在人类偏好上胜过 175B 的 GPT-3——「对齐比规模更重要」的标志性证据；同年 11 月 ChatGPT 上线。' },
  { year:2022, view:'train', name:'Chinchilla',
    paper:'Training Compute-Optimal Large Language Models (Hoffmann et al.)',
    contrib:'参数与数据应等比扩展（70B ↔ 1.4T token）',
    highlights:['scaling'],
    note:'修正 Kaplan 定律的训练配方：同期大多数模型都「数据喂少了」——直接改写此后所有旗舰模型的训练预算分配。' },
  { year:2022, view:'attn', name:'FlashAttention',
    paper:'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al.)',
    contrib:'分块在线 softmax：精确注意力显存 O(L²)→O(L)',
    highlights:['flash'],
    note:'不改数学只改访存，成为全行业默认内核（v2 2023、v3 2024 持续进化）。长上下文训练由此平民化。' },
  { year:2022, view:'ssm', name:'RWKV',
    paper:'RWKV: Reinventing RNNs for the Transformer Era (Peng et al.)',
    contrib:'通道级衰减：训练并行、推理 O(1) 状态的 RNN',
    highlights:['lin','rwkv'],
    note:'循环复兴的社区旗手：把注意力改写为可并行可递推的线性形式。' },
  { year:2023, view:'dense', name:'LLaMA（现代 Dense 范式）',
    paper:'LLaMA: Open and Efficient Foundation Language Models (Touvron et al.)',
    contrib:'RMSNorm + RoPE + SwiGLU 三件套，开源生态引爆',
    highlights:['tok','emb','rope','rms1','swiglu','lnf'],
    note:'现代 Decoder-Only Dense 的定型：Pre-Norm/RMSNorm、旋转位置编码、门控 FFN——Qwen 等后续 Dense 模型全部沿用。' },
  { year:2023, view:'attn', name:'GQA',
    paper:'GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al.)',
    contrib:'分组共享 KV：质量与 Cache 的甜点（LLaMA-2 70B 起标配）',
    highlights:['mqa','gqa'],
    note:'MQA 与 MHA 的折中，还能从旧检查点 Uptraining 转换。Qwen2/3 全系采用。' },
  { year:2023, view:'ssm', name:'Mamba',
    paper:'Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao)',
    contrib:'选择机制 + 硬件感知扫描，线性复杂度比肩 Transformer',
    highlights:['s4','mamba'],
    note:'SSM 高光时刻：参数随输入变化让模型「学会遗忘」；Mamba-2 (2024) 与线性注意力在 SSD 框架下统一。' },
  { year:2023, view:'moe', name:'Mixtral 8x7B',
    paper:'Mixtral of Experts (Jiang et al.)',
    contrib:'开源 Top-2 稀疏 MoE：47B 总参 / 13B 激活',
    highlights:['router','exp1','exp2','exp3','exp4','wsum'],
    note:'MoE 走向主流开源：同等推理成本下质量直逼更大 Dense 模型——此后新旗舰多为 MoE。' },
  { year:2023, view:'attn', name:'Mistral 7B（SWA）',
    paper:'Mistral 7B (Jiang et al.)',
    contrib:'滑动窗口注意力：长文本推理 Cache O(L)→O(w)',
    highlights:['swa','flash'],
    note:'GQA+SWA 组合让 7B 小模型也能跑长文；StreamingLLM 的 attention sinks 解释了首 token 的「注意力黑洞」现象。' },
  { year:2023, view:'train', name:'DPO 直接偏好优化',
    paper:'Direct Preference Optimization (Rafailov et al.)',
    contrib:'免奖励模型、免 RL 采样的偏好对齐',
    highlights:['dpo','rlhf'],
    note:'把 RLHF 两步折叠为一步分类式损失：开源社区的对齐门槛骤降，此后 RLHF/DPO 并存互补。' },
  { year:2024, view:'frontier', name:'TTT 层 / 记忆层',
    paper:'Learning to (Learn at Test Time) (Sun et al.) · Memory Layers at Scale (Meta)',
    contrib:'隐藏状态=在线学习器；稀疏键值记忆替换 FFN',
    highlights:['ttt','memlayer'],
    note:'两条「记忆」路线同时起步：TTT 把状态变成可学习的网络，Memory Layers 把事实知识外置成可检索参数。' },
  { year:2024, view:'frontier', name:'BLT 字节级 LM',
    paper:'Byte Latent Transformer: Patches Scale Better Than Tokens (Meta)',
    contrib:'熵驱动动态 patch，去 BPE 分词器',
    highlights:['blt'],
    note:'计算量随信息熵分配：越难预测的字节花越多算力；等算力下鲁棒性超过 Llama 3——「分词器是否必要」的正面挑战。' },
  { year:2024, view:'gen', name:'MTP 多 token 预测',
    paper:'DeepSeek-V3 Technical Report (DeepSeek-AI)',
    contrib:'辅助头一次预测多步：训练信号更密 + 投机草稿',
    highlights:['autoreg','mtp'],
    note:'多基准 +1.5~3%；推理时 MTP 头直接当 EAGLE 式草稿模型——训练目标与推理加速一鱼两吃。' },
  { year:2024, view:'attn', name:'DeepSeek-V2/V3（MLA）',
    paper:'DeepSeek-V2 / DeepSeek-V3 Technical Report (DeepSeek-AI)',
    contrib:'MLA 潜在压缩 KV Cache 数十倍 + MoE 671B/37B 激活',
    highlights:['gqa','mla','flash'],
    note:'KV 压缩线的当前终点：Cache 只存低秩潜在向量；V3 再加无辅助损失均衡与 MTP 训练目标。' },
  { year:2024, view:'ssm', name:'xLSTM / Mamba-2',
    paper:'xLSTM: Extended Long Short-Term Memory (Beck et al.) · Transformers are SSMs (Dao & Gu)',
    contrib:'指数门控矩阵记忆；SSM 与注意力统一为 SSD',
    highlights:['xlstm','mamba'],
    note:'LSTM 原作者 27 年后的回应；理论层面线性注意力与 SSM 合流——「混合架构」时代开启。' },
  { year:2025, view:'gen', name:'扩散语言模型',
    paper:'LLaDA: Large Language Diffusion Models · Mercury / Gemini Diffusion (Inception Labs)',
    contrib:'并行去噪生成，正面挑战自回归范式',
    highlights:['diffusion','autoreg','spec'],
    note:'LLaDA 8B 首次验证扩散 LM 规模法则；Mercury 主打 5-10× 生成速度——2017 年以来生成范式的第一次真正分叉。' },
  { year:2025, view:'attn', name:'Gemma 3（稳定化）',
    paper:'Gemma 3 Technical Report (Google)',
    contrib:'QK-Norm + 门控注意力 + 5:1 局部/全局 + 多模态',
    highlights:['qknorm','swa'],
    note:'训练稳定化三件套（QK-Norm/soft-cap/z-loss）成为百B级训练的隐形基石——不改变数学，只保证「训得动」。' },
  { year:2025, view:'train', name:'DeepSeek-R1（可验证奖励 RL）',
    paper:'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek-AI)',
    contrib:'纯 RL（GRPO）自发涌现长思维链与反思',
    highlights:['rlvr','rlhf','dpo'],
    note:'「会推理」第一次被证明可以从可验证奖励的强化学习中长出来，而非只靠监督示范——训练范式的新分水岭。' },
  { year:2025, view:'frontier', name:'Titans / 潜空间推理',
    paper:'Titans: Learning to Memorize at Test Time (Google) · Reasoning by Latent Space (Huginnet)',
    contrib:'测试时神经记忆；深度循环潜推理',
    highlights:['titans','looped','ttt'],
    note:'「思考」从输出 token 转移到架构内部：记忆在测试时继续学习、深度按难度自适应——与 o1 式长思维链互补的两条路。' },
  { year:2025, view:'moe', name:'Qwen3 / Kimi K2',
    paper:'Qwen3 Technical Report (Alibaba) · Kimi K2: Open Agentic Intelligence (Moonshot AI)',
    contrib:'细粒度 MoE 成主流：Qwen3-235B-A22B（128 选 8）；K2 1T 总参 / 32B 激活（MLA）',
    highlights:['gqa','router','exp1','exp4'],
    note:'当前开源旗舰的两种答案：Qwen3 同款双模式（思考/非思考）；K2 用 MoE+MLA 把激活参数压到 3%。' },
  { year:2025, view:'attn', name:'NSA / MoBA 稀疏注意力',
    paper:'Native Sparse Attention (DeepSeek-AI) · MoBA: Mixture of Block Attention (Moonshot AI)',
    contrib:'可训练的块稀疏注意力，长上下文成本近线性',
    highlights:['flash','sparse','linear'],
    note:'精确注意力的下一站：只算重要块且稀疏模式进预训练——DeepSeek 与 Moonshot 同月交出答卷。' },
  { year:2025, view:'ssm', name:'Kimi Linear（KDA）',
    paper:'Kimi Linear: An Expressive, Efficient Attention Architecture (Moonshot AI)',
    contrib:'门控线性注意力 3/4 层 + MLA 1/4 层混合，百万级上下文',
    highlights:['kda','lin','mamba','mla'],
    note:'混合架构的代表作：线性层管效率、全注意力管精确检索——「循环复兴」与「注意力演进」两条线的合流点。' },
  { year:2025, view:'frontier', name:'Energy-Based Transformers',
    paper:'Energy-Based Transformers are Scalable Learners (2025)',
    contrib:'能量函数 + 测试时能量下降做隐式推理',
    highlights:['ebt','ttt'],
    note:'把「深思」做进架构：推理 = 能量最小化迭代。与 TTT/Titans 同属测试时计算路线的最新分支，结论尚待大规模验证。' },
  { year:2026, view:'attn', name:'DeepSeek V4（CSA + mHC）',
    paper:'DeepSeek-V4 Technical Report (DeepSeek-AI)',
    contrib:'压缩稀疏注意力：1M 上下文 KV Cache 降至 V3.2 约 10%',
    highlights:['sparse','mla','flash'],
    note:'1.6T 总参/49B 激活（MIT 开源）+ mHC 超连接改进残差通路；三档推理模式把「思考深度」做成 API 参数。' },
  { year:2026, view:'ssm', name:'Kimi K3（2.8T 全栈革新）',
    paper:'Kimi K3: Open Frontier Intelligence (Moonshot AI)',
    contrib:'KDA+MLA 混合 · LatentMoE · Attention Residuals · 全栈 NoPE',
    highlights:['kda','mla','lin'],
    note:'首个开源 3T 级模型（896 专家选 16）+ 首个全 NoPE 前沿模型 + 原生多模态——Kimi Linear 论文的 2.8T 生产化。' },
  { year:2026, view:'attn', name:'推理效率军备竞赛',
    paper:'DeepSeek V4 · Kimi K3 · Nemotron 3 · Qwen3.8（2026 开源旗舰潮）',
    contrib:'组件级效率替换：MoE→LatentMoE、注意力→MLA+线性混合、RoPE→NoPE',
    highlights:['flash','mla','sparse','linear'],
    note:'2026 主线不再是「堆参数」而是「每个组件都换成推理效率调优版」：发布周期压缩至 6-8 周，前沿能力全面开源（MIT/Apache）。' }
];


/* 注册到全局，供 index.html 主脚本使用 */
window.CONTENT_PACK = { MODULE_DETAILS, HISTORY, ERAS, ARCH_NODES, ARCH_FRAMES, ARCH_EDGES,
  ARCH_SPECIAL_EDGES, ARCH_RESIDUALS, ARCH_VIEW_DAWN, ARCH_VIEW_PRE, ARCH_VIEW_DENSE,
  ARCH_VIEW_MOE, ARCH_VIEW_ATTN, ARCH_VIEW_SSM, ARCH_VIEW_GEN, ARCH_VIEW_FRONTIER,
  ARCH_VIEW_TRAIN, ARCH_VIEWS };
})();
