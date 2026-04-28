"""
简单推理示例（单卡 / 调试用）

用法示例：
python examples/simple_inference.py --model-path PATH_TO_WEIGHTS_DIR --prompt "print(1+1)"

注意：真实运行需要加载权重，可能耗费大量显存。示例用于展示调用方式。
"""
import argparse
import torch
from transformers import AutoTokenizer
from model.CustomQwen32B_hybrid import create_hybrid_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='路径到预训练权重目录')
    parser.add_argument('--prompt', type=str, default='Write a Python function to add two numbers.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 仅用于演示：可以指定少量替换层以减小资源占用
    model = create_hybrid_model(model_path=args.model_path, replace_layers=[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
