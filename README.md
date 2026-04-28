# THeWakeSystems-QRUN-Qwen2.5-coder-32B

简要说明：本仓库包含基于 Qwen2.5-Coder-32B 的 Hybrid Q-RUN 版本（在模型中替换部分层为量子层），用于探索参数极限压缩与多卡协作推理方案。

核心信息
- 模型：Qwen2.5-Coder-32B（Hybrid Q-RUN）
- 参数压缩：原始约 3398M → 压缩后 43.7M（约 1.3%）
- 主要用途：数学/逻辑推理、代码生成（注意：代码生成当前存在退化问题）

已知限制
- 代码生成 / 常识 / 多语言任务存在重复 token 与语义断裂问题，可能导致生成片段重复或上下文中断。请谨慎在生产环境使用。

硬件要求
- 推荐：16 卡 CUDA（BF16，总显存约 58.8GB），也支持分布式部署策略。

快速开始
1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 下载权重并放置到 `checkpoints/`（请参考 `checkpoints/README.md`）

3. 运行示例：

```bash
python scripts/benchmark_hybrid.py --model-path checkpoints/checkpoints_hybrid_v2
```

项目结构（已整理）
```
THeWakeSystems-QRUN-Qwen2.5-coder-32B/
├── README.md
├── LICENSE
├── requirements.txt
├── MODEL_CARD.md
├── model/                       # 模型定义与加载代码
├── scripts/                     # 推理、微调、benchmark 脚本
├── examples/                    # 快速上手示例
└── checkpoints/                 # 权重文件说明（不随仓库上传）
```

更多信息见 `MODEL_CARD.md` 与 `checkpoints/README.md`。
