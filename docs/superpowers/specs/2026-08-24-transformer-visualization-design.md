# Transformer 架构交互式可视化 — 设计决策文档

日期：2026-08-24
基准规格：`docs/spec.md`（用户提供，作为完整设计基准，本文档只记录其留白处的实现层决策）

## 决策记录

| # | 决策 | 内容 | 理由 |
|---|------|------|------|
| D1 | 交付形式 | 项目根目录单个 `index.html`，CSS/JS 全内嵌，UI 文案全中文 | spec「交付物要求」 |
| D2 | 计算引擎 | TensorFlow.js v4.x（CDN），权重随机初始化（Glorot/Xavier），做**前向推理模拟**，非预训练模型 | spec 未提供权重文件；教学目的是结构演示 |
| D3 | 图表绘制 | **D3.js v7**（CDN）绘制架构图（SVG）与注意力热力图；位置编码波形用 SVG path | 用户确认采用 spec 技术栈表的 D3 方案 |
| D4 | 公式渲染 | KaTeX（CDN）渲染详情面板中的 LaTeX 公式 | spec 要求 LaTeX 格式公式，KaTeX 比 MathJax 轻且同步渲染 |
| D5 | 分词策略 | 预设句使用内置词典精确分词，保证示例句 `"我 爱 自然语言处理"` 切为 5 个 token（我/爱/自然/语言/处理），与 spec 数据示例一致；任意输入回退为：按空白切分 → 无空格的连续中文按单字切分、连续英文/数字按词切分 | spec 数据示例明确给出 `[5]` 个 token |
| D6 | 可选功能范围 | **全部实现**：深/浅主题切换、tfjs 代码导出（一键复制）、Decoder 掩码自注意力+交叉注意力逐词演示 | 用户确认 |
| D7 | 内存管理 | 所有张量计算包裹在 `tf.tidy()` 中，跨步骤保留的张量显式 `tf.dispose()`，页面卸载时 `tf.disposeVariables()` | spec「控制台无报错，内存管理正常」 |
| D8 | Decoder 演示语义 | 随机权重下生成的文本无语言意义，演示目标是**机制可见**：掩码下三角、逐步 logit/softmax 分布、cross-attention 热力图 | 避免误导用户以为是真实语言模型 |

## 架构模块（沿用 spec 模块划分）

```
Config          配置管理（d_model / num_heads / num_layers / d_ff / dropout_rate / vocab）
Tokenizer       分词器（词典 + 回退规则）
Transformer     tfjs 算子：PositionalEncoding / ScaledDotProductAttention / MultiHeadAttention /
                EncoderLayer / DecoderLayer / Transformer
ParamCounter    参数量统计（随超参数实时更新）
ArchitectureRenderer  D3/SVG 绘制 Encoder-Decoder 结构图，支持悬停高亮与点击选中
Timeline        左侧历史时间轴（2000→2020 六节点），点击联动右侧高亮
AttentionHeatmap      D3 绘制热力图，支持头选择、softmax 前后分数切换
DataFlowAnimator      右面板：分步执行前向传播，日志输出形状/min/max/mean/耗时，当前模块高亮
CodeExporter    根据当前配置生成 tfjs 模型构建代码
Theme           深/浅主题切换（CSS 变量）
App             主控制器：事件绑定、状态管理
```

## 前向传播步骤序列（右面板驱动）

1. 分词 → token 列表
2. Embedding 查表 → `[1, seq_len, d_model]`
3. + Positional Encoding → 波形图 + 形状不变
4. 每层 Encoder：MHA（记录 softmax 前/后分数）→ Add & Norm → FFN → Add & Norm
5. 最终输出表示 `[1, seq_len, d_model]`
6. （可选按钮）Decoder：目标输入右移一位 → 掩码 MHA → cross-MHA（Q=decoder, K/V=encoder 输出）→ 输出层 logits → 下一个 token 概率分布

## 验证标准

- 浏览器打开 `index.html` 控制台无报错
- 默认配置下点击「开始前向传播」可完整走完步骤并渲染热力图
- 切换超参数后张量形状与参数量即时更新
- 1366×768 分辨率三栏布局不溢出，字体 ≥14px
