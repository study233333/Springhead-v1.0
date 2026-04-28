Examples
--------

1. `simple_inference.py` - 基本的单卡推理示例。用法示例：

```bash
python examples/simple_inference.py --model-path PATH_TO_WEIGHTS_DIR --prompt "print(1+1)"
```

备注：示例侧重示范 API 调用方式，真实加载完整权重需要相应显存与多卡支持。
