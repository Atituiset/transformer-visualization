# Transformer 可视化单页应用 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 单文件 `index.html` 实现 Transformer 架构交互式学习应用（结构解剖图 + tfjs 前向传播数据流 + 历史时间轴 + 注意力热力图 + 解码器演示）。

**Architecture:** 零依赖主框架的单 HTML 文件；JS 按 spec 划分为 Config / Tokenizer / tfjs 算子类 / 渲染器类 / 主控制器。所有前向计算用 `tf.tidy()` 包裹，动画回放基于一次性计算的快照数据（不持有张量）。

**Tech Stack:** TensorFlow.js 4.x（CDN，矩阵运算）、D3.js 7（SVG 架构图与热力图）、KaTeX 0.16（LaTeX 公式）、原生 HTML/CSS/JS。验证用全局 playwright CLI（chromium 已缓存）。

**基准文档:** 规格 `docs/spec.md`；决策 `docs/superpowers/specs/2026-08-24-transformer-visualization-design.md`（D1–D8）。

---

## 全局约定（每个任务都遵守）

- **唯一交付物** `/home/atituiset/Projects/transformer-visualization/index.html`。CSS 在 `<style>`，JS 在 `<script>`，按模块注释分节：
  ```
  // ============================================================
  // 模块N：<名称>
  // ============================================================
  ```
- **CDN 固定版本**（`<head>` 中按序加载，均加 `crossorigin="anonymous"`）：
  ```html
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  ```
- **配色（spec 强制）**：主色 `#1a237e`、强调 `#ff6f00`、浅底 `#f5f7fa`；热力图蓝-白-红发散色。深色主题用 CSS 变量覆盖（见 Task 1）。
- **内存纪律**：临时张量一律在 `tf.tidy()` 内创建；需要跨步骤保留的数据在 tidy 内 `.arraySync()`/`.dataSync()` 转为普通数组后带出；权重变量只在模型重建函数中创建并在重建前 `dispose()`。任何任务完成后运行内存检查（Task 14 汇总）。
- **注释**：关键函数 JSDoc（`@param`/`@returns`），中文说明。
- **验证方式**：每个任务用 playwright CLI 打开
  `file:///home/atituiset/Projects/transformer-visualization/index.html`
  （若 CLI 不可用则退化为手动打开），收集 console error（必须为 0），并执行任务给出的 JS 断言表达式。断言统一通过页面上下文 `eval` 执行，输出必须与「预期」逐字匹配。
- **提交**：每任务完成即 `git add index.html docs/ && git commit`（仓库在 Task 1 初始化）。消息格式 `feat(scope): 描述`。

---

### Task 1: 项目脚手架 — 三栏布局骨架 + 主题变量

**Files:**
- Create: `index.html`（完整 HTML 结构 + 全部 CSS）
- Create: `.gitignore`（`*.log`）

- [ ] **Step 1.1: git init**

```bash
cd /home/atituiset/Projects/transformer-visualization && git init -b main
```

- [ ] **Step 1.2: 写入 index.html 骨架**

HTML 结构（body 直接子级）：

```html
<!-- 文件头注释：作者 atituiset / 日期 2026-08-24 / 版本 v1.0.0 -->
<header id="topbar">
  <h1>Transformer 架构交互式可视化</h1>
  <span id="version-tag">v1.0.0</span>
  <button id="theme-toggle">🌙 深色模式</button>
</header>
<main id="layout">
  <aside id="panel-timeline"><h2>历史演进</h2><ol id="timeline-list"></ol></aside>
  <section id="panel-arch">
    <svg id="arch-svg"></svg>
    <div id="detail-panel"></div>   <!-- 详情面板：标题/KaTeX公式容器/形状流/代码/参数量 -->
  </section>
  <aside id="panel-flow">
    <h2>数据流演示</h2>
    <textarea id="input-text">我 爱 自然语言处理</textarea>
    <select id="preset-select"></select>
    <fieldset id="hyper-panel">…d_model/num_heads/num_layers/dropout_rate 各自 select…</fieldset>
    <button id="run-forward">▶ 开始前向传播</button>
    <button id="run-decoder">⏩ Decoder 逐词演示</button>
    <ol id="step-log"></ol>
    <div id="tensor-tracker"></div>
  </aside>
</main>
<div id="chart-area">
  <div id="pe-wave"></div>       <!-- 位置编码波形 -->
  <div id="attn-heatmap"></div>  <!-- 注意力热力图 + 头选择 + pre/post 切换 -->
</div>
```

CSS 要点：`:root{ --c-primary:#1a237e; --c-accent:#ff6f00; --c-bg:#f5f7fa; --c-text:#1c2430; --c-card:#ffffff; }`，`html[data-theme="dark"]{ --c-bg:#10141f; --c-card:#1a2033; --c-text:#e8ecf4; …}` 全部颜色走变量。`#layout{display:grid; grid-template-columns:20% 55% 25%; min-height:calc(100vh - 52px); gap:12px; padding:12px;}`，媒体查询 `@media (max-width:1366px)` 收窄 gap 与字号但不小于 14px。热力图区放中央面板下方（`#chart-area` 并入 `#panel-arch` 内部更合理——采用并入方案：`#detail-panel` 上方依次放 `#pe-wave`、`#attn-heatmap`）。

- [ ] **Step 1.3: 浏览器冒烟验证**

打开页面 → console error 数 = 0；`getComputedStyle(document.body).getPropertyValue('--c-primary').trim()` → `"#1a237e"`；点击 `#theme-toggle` 后 `document.documentElement.dataset.theme` → `"dark"`，再点恢复 `""`/`"light"`；窗口 1366×768 下 `document.documentElement.scrollWidth <= window.innerWidth`。

- [ ] **Step 1.4: Commit** `feat(scaffold): 三栏布局骨架与主题变量`

---

### Task 2: Config 模块 + ParamCounter + 超参数面板

**Files:** Modify `index.html`（新增「模块1：配置管理」「模块2：参数统计」两节 JS + 接线）

- [ ] **Step 2.1: Config 与 ParamCounter**

```js
const Config = { d_model:512, num_heads:8, num_layers:6, d_ff:2048, dropout_rate:0.1 };
// d_ff 恒等于 4*d_model（spec 示例 512→2048）；改 d_model 时同步重算

/** 闭式参数量公式（含偏置）：
 *  Embedding: V*d_model（V=vocab.size）
 *  每层 Encoder/Decoder 层内 MHA: 4*(d²+d)；FFN: 2*d*f+f+d；LayerNorm×2: 4d
 *  输出投影(仅Decoder演示用): V*d+V */
function countParams(cfg, vocabSize){
  const {d_model:d, d_ff:f, num_layers:L} = cfg;
  const perLayer = 4*(d*d+d) + (2*d*f+f+d) + 4*d;
  return { embedding: vocabSize*d, perLayer, encoderTotal: L*perLayer,
           outputProj: vocabSize*d+vocabSize,
           total: vocabSize*d + L*perLayer };   // total 默认只算 Encoder 部分
}
```

超参数面板四个 `<select>`（值按 spec：512/256/128、8/4/2、6/3/1、0.1/0.0），change 事件更新 Config → 调用 `App.onConfigChange()`（本任务先实现为仅刷新参数量文本 + 触发 `window.dispatchEvent(new CustomEvent('config-changed'))`）。

- [ ] **Step 2.2: 页面断言**

默认配置、词表大小 V（由 Task 3 的 Tokenizer 提供；本任务暂以常量 `window.__VOCAB_PLACEHOLDER__=40` 代入）：`countParams(Config,40)` → `{embedding:20480, perLayer:3152384, encoderTotal:18914304, outputProj:20880, total:18934784}`；把 d_model 改成 128 后 `#param-total` 文本包含 `countParams` 新值。

- [ ] **Step 2.3: Commit** `feat(config): 配置管理与闭式参数量统计`

---

### Task 3: Tokenizer（词典分词 + 回退规则）

**Files:** Modify `index.html`（「模块3：分词器」节）

- [ ] **Step 3.1: 实现**

```js
class Tokenizer {
  constructor(){
    /** 多字词词典：贪婪最长匹配；未命中 CJK 单字成 token；[A-Za-z0-9]+ 连续段为一个 token */
    this.lexicon = new Map([
      ['自然语言处理',['自然','语言','处理']], ['机器学习',['机器','学习']],
      ['深度学习',['深度','学习']], ['神经网络',['神经','网络']],
      ['注意力机制',['注意力','机制']], ['人工智能',['人工','智能']],
      ['transformer',['transformer']],
    ]);
  }
  /** @returns {string[]} tokens */
  tokenize(text){ /* 1) 按空白切 segment；2) 每 segment 内贪婪最长词典匹配；
                     3) 未匹配连续 CJK 单字切分；4) 连续 ASCII 字母数字合为一个 token */ }
  getPresets(){ return ['我 爱 自然语言处理','Transformer 是 深度学习 模型',
                        '注意力机制 很 强大','Hello Transformer']; }
  /** 词表：词典全部目标词 + 常用单字池 + 特殊token [<pad>,<bos>,<eos>]，去重定容 */
  buildVocab(){ /* 返回 string[]，写入 window.Vocab */ }
}
```

预设句 `"我 爱 自然语言处理"` 必须产出恰好 `['我','爱','自然','语言','处理']`（设计 D5）。右面板渲染 token 芯片（输入变化时实时刷新）。

- [ ] **Step 3.2: 页面断言**

`new Tokenizer().tokenize('我 爱 自然语言处理')` → `'我,爱,自然,语言,处理'`（长度 5）；`tokenize('Transformer是深度学习')` → 含 `'深度','学习'` 且不含空格 token；`buildVocab()` 后 `countParams(Config, Vocab.length).embedding === Vocab.length*512`。Task 2 的占位符此时替换为真实 `Vocab.length`。

- [ ] **Step 3.3: Commit** `feat(tokenizer): 词典分词与词表构建`

---

### Task 4: PositionalEncoding + 波形图

**Files:** Modify `index.html`（「模块4：位置编码」节 + `#pe-wave` 渲染）

- [ ] **Step 4.1: 实现**

```js
class PositionalEncoding {
  /** @returns {tf.Tensor} [1, maxLen, dModel]，sin偶维/cos奇维，PE(p,2i)=sin(p/10000^{2i/d}) */
  static encode(maxLen, dModel){
    return tf.tidy(()=>{
      const pos=tf.range(0,maxLen).toFloat().reshape([maxLen,1]);
      const i=tf.range(0,dModel).toFloat().reshape([1,dModel]);
      const ang=pos.mul(tf.pow(10000, i.div(dModel).floor().mul(2).div(dModel))); // 2i/d
      return tf.concat([ang.sin(), ang.cos()],1).mul(tf.scalar(0))
               .add(this._interleave(ang.sin(),ang.cos())).reshape([1,maxLen,dModel]);
    });
  }
}
```
（实现时允许改用纯 JS 双重循环填充 `Float32Array` 再 `tf.tensor3d` —— 更直观且无拼接技巧负担；上式仅为数学规格说明，**以公式为准**：even=sin(pos/10000^(2i/d))，odd 同式取 cos。）

波形图：取 `pos∈[0,seq_len)` 的 8 条固定维度曲线（i=0,1,4,5,16,17,64,65 截到 d_model 内）画 SVG path，X 轴=pos，Y 轴=[-1,1]，标注维度号；用 D3 或手写 path 均可（本组件手写 SVG polyline，D3 留给热力图）。

- [ ] **Step 4.2: 页面数值断言**

```js
const pe = PositionalEncoding.encode(5,8).arraySync()[0];
JSON.stringify([pe[0][0],pe[0][1],Math.round(pe[1][0]*1e4)/1e4,Math.round(pe[1][1]*1e4)/1e4])
```
→ `"[0,1,0.8415,0.5403]"`（sin(0)=0、cos(0)=1、sin(1)、cos(1)）；波形图 `#pe-wave svg path` 数量 ≥8。

- [ ] **Step 4.3: Commit** `feat(pe): 位置编码算子与波形可视化`

---

### Task 5: ScaledDotProductAttention + MultiHeadAttention + 掩码

**Files:** Modify `index.html`（「模块5：核心算子」节）

- [ ] **Step 5.1: 实现类**

```js
/** @returns {{output, pre:[H,L,L]数组的普通数组, post:同}} pre=缩放后未mask分数，post=mask后softmax概率 */
class ScaledDotProductAttention {
  forward(Q,K,V,maskAdd=null){ /* scores=QKᵀ/√dk → masked=scores+maskAdd → w=softmax(masked,-1)
        返回 output=wV；pre/post 均 arraySync() 快照（形状 [B,H,L,L] 取 batch0 → [H,L,L]）*/ }
}
function causalMask(L){ /* tf [1,1,L,L]，下三角0其余-1e9 */ }

class MultiHeadAttention {
  constructor(dModel,numHeads){ /* WQ,WK,WV,WO: tf.variable(glorotInit([d,d])) + bQ..bO
                                   dh=d/numHeads；glorotInit=sqrt(2/(fanIn+fanOut))*randomNormal */ }
  splitHeads(x){/* [B,L,D]→transpose→reshape [B,H,L,dh] */}
  combineHeads(x){/* 逆操作 */}
  forward(Qin,Kin,Vin,maskAdd){ /* 投影→split→SDPA→combine→WO+bO
        返回 {output:[B,L,D](张量), weights:{pre,post}[H,L,L]} */}
}
```

- [ ] **Step 5.2: 页面数值断言**

① 无掩码行归一：`mha.forward(x,x,x,null)` 后 `post[h]` 每行和 ≈1（误差<1e-5）。
② 因果掩码：L=3 时 `post[任意h]` 第 0 行 → `[1,0,0]`（第 2 位 <1e-6）；第 2 行和≈1。
③ 形状链：d_model=16,h=4,输入[1,5,16] → `output.shape` = `[1,5,16]`，`post.length`=4。
④ 确定性：同输入两次 forward（不重建权重）输出 `allClose`（<1e-6）。

- [ ] **Step 5.3: Commit** `feat(attn): 缩放点积注意力与多头注意力`

---

### Task 6: LayerNorm + EncoderLayer/DecoderLayer + Transformer 编排

**Files:** Modify `index.html`（「模块6：层级编排」节）

- [ ] **Step 6.1: 实现**

```js
class LayerNorm { /* gamma=tf.variable(ones[d]) beta=zeros[d]; normalize(x): moments(-1) */ }
class FeedForward { /* W1[f,d] b1[f] W2[d,f] b2[d]; relu→linear; 记录中间激活形状供日志 */ }
class EncoderLayer { forward(x){ /* x=LN(x+Dropout(MHA(x))) → LN(x+Dropout(FFN(x)))
                                     training=false 时 Dropout 为恒等；返回{output,attn} */ } }
class DecoderLayer { forward(x,encOut,selfMask,crossMaskNull){
  /* 掩码MHA → LN(add) → cross-MHA(Q=x,K=V=encOut) → LN(add) → FFN → LN(add);
     返回 {output, selfAttn, crossAttn} */ } }
class Transformer {
  build(cfg){ /* 先 dispose 旧变量再建 L 层 EncoderLayer + L 层 DecoderLayer + 输出投影 W_out[V,d] */}
  encode(tokenIds){ /* embedding查表(tf.gather)×√d_model + PE → 逐层 → 返回
     {steps:[每层快照], final:[1,L,d]数组} 全部在 tidy 内转普通数组 */ }
  decodeStep(targetIds, encOutArr){ /* 目标右移(<bos>开头)→掩码MHA→cross→FFN→logits=W_out·x
     → softmax 概率数组返回 */ }
}
```

Embedding 表 `E[V,d]` 为 variable；`tf.gather(E, ids)` 后乘 `√d_model`（原论文做法，详情面板注明）。

- [ ] **Step 6.2: 页面端到端断言**

默认配置 + 5 个 token：`app.transformer.encode(ids)` → `final` 形状 `[1][5][512]`、无 NaN；`decodeStep` 返回长度 `Vocab.length` 的概率数组且和≈1；连续调用两次 forward 后 `tf.memory().numTensors` 与调用前差值 ≤ 变量数（tidy 未泄漏临时张量）。

- [ ] **Step 6.3: Commit** `feat(model): Encoder/Decoder 层堆叠与前向编排`

---

### Task 7: 模块详情注册表 + KaTeX 详情面板

**Files:** Modify `index.html`（「模块7：详情内容」节）

- [ ] **Step 7.1: 内容表**

```js
const MODULE_DETAILS = {
  embedding:{ title:'输入嵌入层',
    latex:['\\text{Emb}(x)=E[x]+\\text{PE}','E\\in\\mathbb{R}^{|V|\\times d_{model}}'],
    shapeFlow:['[seq_len]','[1, seq_len, d_model]'],
    code:`const emb = tf.gather(E, ids).mul(Math.sqrt(dModel));`,
    params:'V × d_model' },
  mha:{ latex:['\\text{Attention}(Q,K,V)=\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
        '\\text{MultiHead}(Q,K,V)=\\text{Concat}(h_1..h_h)W^O'], … },
  addnorm:{ /* 残差+LN 公式 */ }, ffn:{ /* max(0,xW₁+b₁)W₂+b₂ */ },
  pe:{ /* 正弦位置编码两式 */ }, encoder:{}, decoder:{ /* 掩码与交叉注意力公式 */ },
};
```
（spec「关键公式」四组全量录入，字符串内的反斜杠按 JS 字符串规则转义。）选中模块时 `katex.render(tex, el, {throwOnError:false})` 逐条渲染 + 形状流箭头列表 + 代码片段 `<pre>` + 参数量。

- [ ] **Step 7.2: 页面断言**

`Object.keys(MODULE_DETAILS)` ⊇ `['embedding','pe','mha','addnorm','ffn','encoder','decoder']`；程序化选中 `mha` 后 `#detail-panel .katex` 元素数 = 该项 latex 条数，且面板文本含 `[batch, seq_len, d_model]` 类形状串。

- [ ] **Step 7.3: Commit** `feat(detail): 模块公式/代码/形状详情面板`

---

### Task 8: ArchitectureRenderer（D3/SVG 结构图）

**Files:** Modify `index.html`（「模块8：架构渲染器」节）

- [ ] **Step 8.1: 节点声明 + 渲染**

```js
const ARCH_NODES=[
 {id:'input',label:'输入 Tokens',col:'enc'}, {id:'embed',label:'Embedding',col:'enc'},
 {id:'pe',label:'+ 位置编码',col:'enc'},
 {id:'enc-mha',label:'多头自注意力',col:'enc'},{id:'enc-addnorm1',label:'Add & Norm',col:'enc'},
 {id:'enc-ffn',label:'前馈网络',col:'enc'},{id:'enc-addnorm2',label:'Add & Norm',col:'enc'},
 {id:'enc-stack',label:'× N 层',col:'enc',frame:true},
 {id:'dec-input',…},{id:'dec-masked-mha',label:'掩码多头注意力',col:'dec'},
 {id:'dec-cross',label:'交叉注意力\n(Q←解码器, K/V←编码器)',col:'dec'},
 {id:'dec-ffn',…},{id:'dec-linear',label:'Linear + Softmax',col:'dec'},
 {id:'out',label:'输出概率',col:'dec'}];
// Add & Norm 的跳跃连接用从下层绕行的圆角贝塞尔路径表达“残差”
class ArchitectureRenderer{
  render(nodes){ /* D3: g.node(rect+text) + marker-end 箭头连线；残差旁路虚线 */ }
  highlight(ids){ /* 命中节点加 .active（描边 var(--c-accent)+发光滤镜），其余降透明度 */ }
  clearHighlight(){}
}
```
点击节点 → `App.selectModule(id)` → 详情面板（Task 7）+ `highlight([id])`。悬停显示 tooltip（模块一句话职责）。

- [ ] **Step 8.2: 页面断言**

`#arch-svg .node` 数量 ≥ 14；`renderer.highlight(['enc-mha'])` 后 `#arch-svg .node.active` 存在且文本含「多头自注意力」；点击 `#arch-svg` 中 ffn 节点后 `#detail-panel` 标题变为 FFN 相关。

- [ ] **Step 8.3: Commit** `feat(arch): D3 编解码器结构图与选中联动`

---

### Task 9: Timeline 历史时间轴 + 年份联动高亮

**Files:** Modify `index.html`（「模块9：历史时间轴」节）

- [ ] **Step 9.1: 数据 + 渲染 + 联动**

```js
const HISTORY=[
 {year:2000,name:'RNN / LSTM',paper:'Long Short-Term Memory',contrib:'门控循环单元缓解梯度消失',
  highlights:['embed'],note:'顺序处理：t 时刻依赖 t-1，无法并行'},
 {year:2014,name:'Seq2Seq + Attention',paper:'Neural Machine Translation by Jointly Learning to Align and Translate',
  contrib:'编码器-解码器 + 对齐注意力',highlights:['dec-cross'],note:'注意力的起源'},
 {year:2017,name:'Transformer',paper:'Attention Is All You Need',contrib:'完全基于注意力，抛弃循环结构',
  highlights:['embed','pe','enc-mha','enc-ffn','dec-masked-mha','dec-cross']},
 {year:2018,name:'BERT',paper:'BERT: Pre-training of Deep Bidirectional Transformers',
  contrib:'双向编码器预训练',highlights:['embed','enc-mha','enc-ffn','enc-stack'],note:'只用 Encoder'},
 {year:2019,name:'GPT-2',paper:'Language Models are Unsupervised Multitask Learners',
  contrib:'大规模解码器语言建模',highlights:['dec-input','dec-masked-mha','dec-ffn','out'],note:'只用 Decoder'},
 {year:2020,name:'GPT-3',paper:'Language Models are Few-Shot Learners',contrib:'1750亿参数，上下文少样本学习',
  highlights:['dec-input','dec-masked-mha','out']}];
class Timeline{ render(list){ /* 垂直时间线：年份圆点+卡片（论文题/贡献），支持滚动，
    节点间连接线；点击 → renderer.highlight(node.highlights) + 显示 note 于时间轴底部 */ } }
```

- [ ] **Step 9.2: 页面断言**

`#timeline-list li` 数量 = 6；点击 2017 节点后 active 节点数 = `highlights.length`(6) 且 `#timeline-note` 含「抛弃循环结构」；时间轴容器 `scrollHeight > clientHeight`（可滚动）。

- [ ] **Step 9.3: Commit** `feat(timeline): 历史演进时间轴与架构联动`

---

### Task 10: AttentionHeatmap（先读 dataviz 技能再动手）

**Files:** Modify `index.html`（「模块10：注意力热力图」节）

- [ ] **Step 10.0:** 调用 `Skill(dataviz)`——其调色板/对比度规范用于热力图配色；**冲突时以 spec 配色为准**（蓝-白-红发散渐变，如插值 `d3.interpolateRdBu` 反向或自定义 `#1a237e→#fff→#ff6f00`？——采用 spec 字面要求：蓝(低)-白-红(高)，实现为 `d3.interpolateRgbBasis(['#2166ac','#f7f7f7','#b2182b'])`）。

- [ ] **Step 10.1: 组件**

```js
class AttentionHeatmap{
  render({tokens, pre, post}){ /* D3：cells=L×L rect + 行(Query)/列(Key)标签；
     数据存 this.data 供重绘；默认显示 post */ }
  setHead(h){}          // 1..num_heads
  setStage('pre'|'post'){} // pre 显示缩放分数：按当前矩阵 min-max 归一上色，tooltip 显原值
  setMatrix(layerIdx){} // 多层时选层
  clear(){}
}
```
tooltip 显示 `Q="我" K="爱" score=0.123`；行列标签字体 ≥12px、旋转列标签。

- [ ] **Step 10.2: 页面断言**

构造样例 5×5 两头矩阵传入 render → `.cell` 数 =25；`setHead(2)` 后重绘不变数量；`setStage('pre')` 后 tooltip 原值域 ≠ [0,1]（存在负值或 >1），`setStage('post')` 回到概率域；切换头按钮数量 = num_heads。

- [ ] **Step 10.3: Commit** `feat(heatmap): 注意力热力图与头/阶段切换`

---

### Task 11: 前向传播引擎 + DataFlowAnimator 步骤日志

**Files:** Modify `index.html`（「模块11：数据流动画」节 + App 接线）

- [ ] **Step 11.1: 引擎（快照式）+ 动画器**

```js
class ForwardEngine{
  /** 一次 tidy 内跑完，产出纯数据步骤序列 */
  run(text){
    // steps: [{label,moduleId,shapeText,stats:{min,max,mean},ms,extra}]
    // 1 分词 → 2 Embedding(+√d) → 3 +PE(附波形刷新) → 4..每层:
    //   MHA(pre/post 存入 heatmap 缓存, layer=highest?)→AddNorm→FFN(中间[1,L,d_ff])→AddNorm
    // → 最终表示。每步 ms 用 performance.now() 差值。
  }
}
class DataFlowAnimator{
  async play(steps,{onStep,onDone}){ /* setTimeout 链逐步：highlight(moduleId) +
     日志追加行「③ 多头注意力 | [1,5,512] | min -2.31 max 3.10 mean 0.02 | 4.2ms」+
     张量追踪器同步；间隔 450ms 可配 */ }
  stop(){} reset(){}
}
```
日志行格式化函数 `fmtShape([1,5,512])→"[1, 5, 512]"`、`fmtStats` 保留两位小数。运行中再次点击 → 先 stop 重置。运行结束热力图展示**最后一层**注意力，提供层下拉（1..N）。

- [ ] **Step 11.2: 页面断言**

默认句点击 `#run-forward` 后等待完成：`#step-log li` 数 = 3 + 2×6×?（每层 3 步：MHA/AddNorm/FFN+AddNorm 合并为 3 行）+1 ≥ 20 且首行含「分词」；末行形状 `[1, 5, 512]`；每行含 `ms`；`#attn-heatmap .cell`=25；全程 console error=0；连跑两次 `tf.memory().numTensors` 差 ≤ 变量总数。

- [ ] **Step 11.3: Commit** `feat(flow): 前向传播分步动画与张量追踪`

---

### Task 12: Decoder 逐词演示

**Files:** Modify `index.html`（「模块12：解码器演示」节）

- [ ] **Step 12.1: 实现**

点击 `#run-decoder`：目标序列初始化 `['<bos>']`；循环最多 min(源长+3, 8) 步：
1. `decodeStep` 得概率分布 → 高亮 `dec-*` 节点；
2. 掩码自注意力热力图（下三角可见、右上灰显）与 cross 热力图（行=已生成词，列=源词）交替展示于 `#attn-heatmap`（加 tab 切换 self/cross）；
3. Top-5 概率水平条形图（SVG，标注 token 名与百分比）追加到日志区；
4. argnext 取 argmax 作为下一步输入（随机权重下文本无语义——日志区固定注明「演示：权重随机初始化，仅展示生成机制」（设计 D8））。
每步高亮对应 decoder 节点，耗时计入日志。

- [ ] **Step 12.2: 页面断言**

点击后等待完成：日志含 ≥3 个「Top-5」块；self-attention 矩阵第 r 行第 c 列（c>r）单元格透明度为背景级（mask 生效）；概率条形图最大值 ≈ 分布最大概率；console error=0。

- [ ] **Step 12.3: Commit** `feat(decoder): 掩码自注意力与逐词生成演示`

---

### Task 13: CodeExporter + Theme 完善

**Files:** Modify `index.html`（「模块13：代码导出」节）

- [ ] **Step 13.1: 导出**

```js
class CodeExporter{
  generate(cfg,vocabSize){ /* 返回模板字符串：tf.sequential 构建 embedding→PE(自定义层注释)→
     L×(multiHeadAttention 自定义层注释+layerNorm+dense relu d_ff+dense d_model)，
     数值全部来自当前 cfg */ }
}
```
详情面板底部「📋 导出当前配置的模型代码」按钮 → 弹出 `<pre>` + 「复制」按钮（`navigator.clipboard.writeText`，失败回退 `document.execCommand('copy')`），复制成功按钮文案 2 秒变「✅ 已复制」。

- [ ] **Step 13.2: 页面断言**

导出文本含 `dense(${Config.d_ff})` 与 `repeat(${Config.num_layers})` 字样的等价片段（正则断言实际数值出现）；改 d_model=256 后重新导出数值随之改变；复制按钮触发后（clipboard 权限拒绝时走回退分支也不报错）console error=0。

- [ ] **Step 13.3: Commit** `feat(export): tfjs 代码导出与主题持久化`

---

### Task 14: 最终验收（verification-before-completion）

- [ ] **Step 14.1: 全功能回归**（playwright 一次脚本跑完）

① 控制台 error/warning（非资源加载提示）= 0；② 四组超参数各切换一次后再跑前向传播成功且日志形状随 d_model 变化（512→`[1,5,512]`、256→`[1,5,256]`）；③ 时间轴 6 节点逐一点击无异常；④ 深浅主题切换两次；⑤ `tf.memory().numTensors` 在 3 轮前向+2 轮解码前后差值 ≤ 变量总数（无泄漏）；⑥ 文件头部注释含作者/日期/版本。

- [ ] **Step 14.2: 1366×768 截图验收**

视口设为 1366×768：整页截图（浅色）、前向传播运行中截图、深色主题截图。检查：三栏无横向溢出、最小字号 ≥14px、热力图与波形完整可见。截图保存至 `docs/screenshots/`。

- [ ] **Step 14.3: Commit + 收尾** `docs: 验收截图与收尾` ；向用户汇报结果与已知限制（随机权重、CDN 依赖网络）。

---

## Self-Review 记录

- **Spec coverage**：时间轴(§1)→T9；架构解剖+悬停详情(§2)→T7/T8；数据流演示(§3)→T11；注意力可视化(§4)→T5/T10；参数配置面板(§5)→T2；代码导出(§6)→T13；布局/配色/响应式→T1/T14；公式四组→T7；JSDoc/头部注释→全局约定+T13 验收；tf.dispose→T6/T11/T14。无遗漏。
- **Placeholder scan**：无 TBD/TODO；所有类给出签名与行为描述，数学函数给出精确公式或数值断言。
- **Type consistency**：`MODULE_DETAILS` 键与 `ARCH_NODES.id` 对应关系已在 T7/T8 固定（`enc-mha/enc-ffn/dec-cross` 等）；`AttentionHeatmap.render({tokens,pre,post})` 与 T11 引擎产物字段一致；`countParams` 返回键在 T2 断言中使用一致。
