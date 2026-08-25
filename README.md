# Transformer 架构演进交互式可视化

[![Deploy to GitHub Pages](https://github.com/Atituiset/transformer-visualization/actions/workflows/deploy.yml/badge.svg)](https://github.com/Atituiset/transformer-visualization/actions/workflows/deploy.yml)
![Version](https://img.shields.io/badge/version-v2.3.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> 从 1943 年 MP 神经元到 2026 推理效率竞赛——在浏览器里完整走一遍深度学习与大模型的进化之路。

**在线访问：[https://atituiset.github.io/transformer-visualization/](https://atituiset.github.io/transformer-visualization/)**

纯前端单页应用（单 HTML 文件、零构建、零后端），所有前向计算由 TensorFlow.js 在浏览器本地完成。

## 功能

- **10 大架构视图**：前传基石（1943-2016）· 前史 RNN→2017 · 2017 原版 Encoder-Decoder · Decoder-Only Dense · Decoder-Only MoE · Attention 演进 · 线性/SSM · 生成范式 · 前沿探索 · 训练与对齐
- **55+ 里程碑论文线**（1943→2026）：时间轴按四阶段分组，点击节点联动架构图高亮 + 抽屉解释（公式/代码/参数量/论文引用）
- **真实前向传播**：三套 TensorFlow.js 引擎可运行——2017 原版（含 Decoder 逐词生成）、现代 Dense（RMSNorm+RoPE+GQA+SwiGLU）、MoE（Router Top-2 稀疏路由 + 专家负载统计）
- **预训练权重（可选）**：训练台一键加载 TinyShakespeare 字符级模型（`tools/train.html` 浏览器内训练产物，与演示架构逐算子一致），Dense 生成即真实语义
- **在线微调**：对已加载的预训练权重跑真实反向传播（Adam，约 10 秒），loss 曲线实时绘制——可视化中的变量即被训练的变量
- **生成采样控制**：temperature / top-k / top-p 滑杆，Top-K 图表实时展示「截断 + 重归一」后的分布变形
- **KV Cache 增量解码**：Prefill 一次 + 每步仅前向 1 个新 token，解码步热力图 = 新 token 对全缓存的注意力，附整段重算耗时对照
- **单步调试**：底部控制台支持播放/暂停/单步前进/回退/跳转，点击日志任意步骤跳转并弹出解释抽屉
- **注意力热力图**：头/层/Softmax 前后切换 + **Model View**（全层×全头矩阵总览，点击定位）+ **Neuron View**（Q·K 逐维点积分解，点击热力图单元格）
- **代码导出**：按当前超参数生成可运行的 tfjs 模型构建代码，一键复制
- **中英双语（v2.4）**：🌐 一键切换界面语言；界面全量双语；架构图/时间轴深度内容英文包 `content.en.js` 已覆盖标签与条目名，长段解释暂以中文显示（持续补全中）；文档双语（[docs/功能说明.md](docs/功能说明.md) / [docs/guide-en.md](docs/guide-en.md)）

## 知识覆盖（四阶段）

| 阶段 | 内容 |
|------|------|
| 神经网络诞生 1943-1986 | MP 神经元 → Hebb → 感知机 → 反向传播 |
| 深度学习复兴 1990-2016 | RNN/LSTM → Seq2Seq+Attention → AlexNet/Adam/GAN/BN/ResNet |
| Transformer 时代 2017-2022 | 原版架构 → 三大范式 → Dense/MoE/效率注意力/SSM 各线分流 |
| 后 Transformer 2023-2026 | LLaMA 范式 → MLA/FlashAttention/Mamba/KDA → 扩散 LM/投机解码 → DeepSeek V4 / Kimi K3 |

## 本地运行

```bash
# 直接双击 index.html 即可（预训练权重加载需 http 服务），或：
python3 -m http.server 8642
# 访问 http://127.0.0.1:8642/index.html
```

## 复现预训练权重（可选）

```bash
python3 -m http.server 8642
# 浏览器打开 http://127.0.0.1:8642/tools/train.html?steps=2000
# 训练完成后在 DevTools 控制台执行 copy(__export())，粘贴保存为 assets/pretrained-dense.json
```

## 技术栈

原生 HTML/CSS/JS（单文件零构建）· [TensorFlow.js](https://www.tensorflow.org/js) 4.x（前向推理）· [D3.js](https://d3js.org/) 7（架构图/热力图）· [KaTeX](https://katex.org/)（公式渲染）

## 已知限制

- 默认权重为随机初始化：演示展示机制而非语言能力（可在「🎓 训练台」一键加载真实训练权重获得有语义的生成）
- MoE 演示为示意实现（全专家计算后按门控合成；真实推理只计算被选中专家）
- 生成演示已实现 KV Cache 增量解码（Prefill 一次 + 每步仅前向 1 个新 token，含整段重算耗时对照；微型模型上增量耗时不占优属预期，收益随上下文长度放大）；缓存不持久化，每次运行重建
- 预训练权重仅适配 d128/h4/N3 配置，修改超参数后自动失效回退
- 依赖 CDN（tfjs/D3/KaTeX）与 assets/ 权重文件，离线打开需自行内联
