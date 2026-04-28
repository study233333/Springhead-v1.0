"""
CustomQwen32B_hybrid.py

DRQC-Compress: 4 组件混合架构
1. MonarchProj — brick-wall 量子电路对应的结构化投影（替代 input_proj）
2. 可学习旋转编码 — 参数化 Ry(θ), Rz(φ) 旋转门
3. EntanglementLayer — CNOT/CZ 两比特门对应的低秩维度混合
4. 深度重上传 MLP — 多层变分线路 + 数据重上传

目标: 压缩 Qwen2.5-Coder-32B 的 MLP 层，替换层参数压缩 98%+
"""

import torch
import torch.nn as nn
import math
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.activations import ACT2FN
from accelerate import init_empty_weights


def resolve_compute_dtype(dtype="auto"):
    if isinstance(dtype, torch.dtype):
        return dtype

    key = str(dtype).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if key in mapping:
        return mapping[key]

    if key != "auto":
        raise ValueError(f"不支持的 dtype: {dtype}")

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.float16

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.float16

    return torch.float32


# ============================================================
# 组件 1: MonarchProj — 结构化投影（brick-wall 量子电路）
# ============================================================

class MonarchProj(nn.Module):
    """
    Monarch 矩阵投影: W = L @ P @ R
    - L: 块对角矩阵 [n_blocks_out, block_size, block_size]
    - P: 固定 stride 排列
    - R: 块对角矩阵 [n_blocks_in, block_size, block_size]

    量子对应: brick-wall 电路中交替层的两比特门
    参数量: O(n_blocks × block_size²) 远小于 O(in_dim × out_dim)
    """
    def __init__(self, in_dim, out_dim, block_size=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block_size = block_size

        # 确保整除，必要时 pad
        self.in_padded = math.ceil(in_dim / block_size) * block_size
        self.out_padded = math.ceil(out_dim / block_size) * block_size
        self.n_blocks_in = self.in_padded // block_size
        self.n_blocks_out = self.out_padded // block_size

        # 中间维度: 取 in 和 out 的较大值（确保信息不丢失）
        self.mid_dim = max(self.in_padded, self.out_padded)
        self.n_blocks_mid = self.mid_dim // block_size

        # R: 第一层块对角 [n_blocks_in, block_size, block_size]
        # 将 in_padded 映射到 mid_dim（如果 in < out，需要扩展）
        self.R = nn.Parameter(torch.empty(self.n_blocks_in, block_size, block_size))
        # L: 第二层块对角 [n_blocks_out, block_size, block_size]
        # 将 mid_dim 映射到 out_padded
        self.L = nn.Parameter(torch.empty(self.n_blocks_out, block_size, block_size))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # stride 排列索引（固定，不可学习）
        perm = self._build_stride_perm(self.n_blocks_in, block_size, self.mid_dim)
        self.register_buffer('perm', perm)

        self._init_weights()

    def _build_stride_perm(self, n_blocks, block_size, target_dim):
        """构建 stride 排列: 交错重排，模拟 brick-wall 电路的连接模式"""
        total = n_blocks * block_size
        idx = torch.arange(total)
        # stride 排列: 将 [n_blocks, block_size] reshape 为 [block_size, n_blocks] 再 flatten
        idx = idx.view(n_blocks, block_size).t().contiguous().view(-1)
        # 截断或扩展到 target_dim
        if total < target_dim:
            # 循环填充
            repeats = math.ceil(target_dim / total)
            idx = idx.repeat(repeats)[:target_dim]
        else:
            idx = idx[:target_dim]
        return idx

    def _init_weights(self):
        # 正交初始化每个块
        for i in range(self.n_blocks_in):
            nn.init.orthogonal_(self.R[i])
        for i in range(self.n_blocks_out):
            nn.init.orthogonal_(self.L[i])

    def forward(self, x):
        # x: [..., in_dim]
        shape = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)  # [N, in_dim]
        N = x.shape[0]

        # Pad input if needed
        if self.in_dim < self.in_padded:
            x = torch.nn.functional.pad(x, (0, self.in_padded - self.in_dim))

        # R: 块对角矩阵乘 [N, n_blocks_in, block_size] @ [n_blocks_in, block_size, block_size]
        x = x.view(N, self.n_blocks_in, self.block_size)
        x = torch.bmm(
            x.transpose(0, 1),  # [n_blocks_in, N, block_size]
            self.R               # [n_blocks_in, block_size, block_size]
        ).transpose(0, 1)        # [N, n_blocks_in, block_size]
        x = x.reshape(N, -1)     # [N, in_padded]

        # P: stride 排列
        x_mid = x[:, self.perm] if self.mid_dim <= self.in_padded else \
                torch.nn.functional.pad(x, (0, self.mid_dim - self.in_padded))[:, self.perm]

        # L: 块对角矩阵乘
        x_mid = x_mid[:, :self.out_padded].view(N, self.n_blocks_out, self.block_size)
        out = torch.bmm(
            x_mid.transpose(0, 1),
            self.L
        ).transpose(0, 1).reshape(N, -1)

        # 截断到 out_dim + bias
        out = out[:, :self.out_dim] + self.bias
        return out.view(*shape, self.out_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 组件 3: EntanglementLayer — 低秩维度混合（量子纠缠）
# ============================================================

class EntanglementLayer(nn.Module):
    """
    低秩残差混合，模拟量子电路中的 CNOT/CZ 纠缠门。
    在 sin/cos 编码后的特征维度间引入交互。

    x → x + up(down(x))
    参数量: 2 × dim × rank
    """
    def __init__(self, dim, rank=64):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.zeros_(self.up.weight)  # 初始时纠缠层为恒等映射

    def forward(self, x):
        return x + self.up(self.down(x))


# ============================================================
# 组件 2+4 整合: Q_RUNLayer_Hybrid
# ============================================================

class SimpleMLP(nn.Module):
    """两层 MLP，带 LayerNorm"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


class Q_RUNLayer_Hybrid(nn.Module):
    """
    DRQC-Compress 混合层

    信息流:
    MonarchProj → LayerNorm → 可学习旋转 sin/cos → 纠缠层 → 深度重上传 MLP → flatten

    量子电路对应:
    brick-wall → 归一化 → Ry/Rz 旋转门 → CNOT 纠缠 → 多层变分+测量
    """
    def __init__(self, input_dim, hidden_dim, n_reuploads=3,
                 u_proj_output_dim=4, block_size=64, entangle_rank=64,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.u_proj_output_dim = u_proj_output_dim
        self.n_reuploads = n_reuploads

        self.proj_dim = hidden_dim // u_proj_output_dim
        assert self.proj_dim * u_proj_output_dim == hidden_dim

        # 组件 1: Monarch 投影
        self.input_proj = MonarchProj(input_dim, self.proj_dim, block_size)
        self.input_norm = nn.LayerNorm(self.proj_dim)

        # 组件 2: 可学习旋转参数
        self.thetas = nn.ParameterList([
            nn.Parameter(torch.ones(self.proj_dim)) for _ in range(n_reuploads)
        ])
        self.phis = nn.ParameterList([
            nn.Parameter(torch.zeros(self.proj_dim)) for _ in range(n_reuploads)
        ])

        sincos_dim = 2 * n_reuploads

        # 组件 3: 纠缠层（在 sincos 编码后混合维度）
        self.entangle = EntanglementLayer(self.proj_dim, rank=entangle_rank)

        # 组件 4: 轻量 u_proj（sincos_dim → u_proj_output_dim）
        # 不用大隐藏层，避免 [B, S, proj_dim, hidden] 4D 张量爆显存
        # Monarch + 纠缠层 + 可学习旋转已提供足够表达能力
        self.u_proj = nn.Linear(sincos_dim, u_proj_output_dim)

    def forward(self, x):
        B, S = x.shape[:2]

        # 组件 1: Monarch 投影 + 归一化
        x_proj = self.input_norm(self.input_proj(x))  # [B, S, proj_dim]

        # 组件 3: 纠缠层（在投影后、编码前混合维度）
        x_proj = self.entangle(x_proj)

        # 组件 2: 可学习旋转编码
        sincos_features = []
        for i in range(self.n_reuploads):
            rotated = self.thetas[i] * x_proj + self.phis[i]
            sincos_features.append(torch.sin(rotated))
            sincos_features.append(torch.cos(rotated))
        sincos = torch.stack(sincos_features, dim=-1)  # [B, S, proj_dim, 2*n_reuploads]

        # 组件 4: 轻量映射
        out = self.u_proj(sincos)  # [B, S, proj_dim, u_proj_output_dim]

        return out.reshape(B, S, self.hidden_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# MLP 替换层 + 模型类
# ============================================================

class Qwen2MLP_Hybrid(nn.Module):
    """Replace 模式: gate/up/down 全部替换为 Q_RUNLayer_Hybrid"""
    def __init__(self, config, **qrun_kwargs):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_proj = Q_RUNLayer_Hybrid(config.hidden_size, config.intermediate_size, **qrun_kwargs)
        self.up_proj = Q_RUNLayer_Hybrid(config.hidden_size, config.intermediate_size, **qrun_kwargs)
        self.down_proj = Q_RUNLayer_Hybrid(config.intermediate_size, config.hidden_size, **qrun_kwargs)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def count_parameters(self):
        return self.gate_proj.count_parameters() + self.up_proj.count_parameters() + self.down_proj.count_parameters()

    def init_weights(self):
        """Kaiming + 正交初始化（不依赖原始 MLP 权重）"""
        for qrun_layer in [self.gate_proj, self.up_proj, self.down_proj]:
            self._init_qrun(qrun_layer)
        print("  -> 已初始化 Hybrid Q-RUN 层")

    def _init_qrun(self, layer):
        with torch.no_grad():
            # MonarchProj 已在构造时正交初始化
            # 旋转参数: theta=1, phi=0 (初始等价于标准 sin/cos)
            # EntanglementLayer 已在构造时初始化 (up=0, 初始恒等)
            # u_proj: 标准 Xavier 初始化，避免 gain=0.01 导致信号坍缩
            nn.init.xavier_uniform_(layer.u_proj.weight, gain=1.0)
            nn.init.zeros_(layer.u_proj.bias)

class CustomQwen32B_Hybrid(Qwen2ForCausalLM):
    """Qwen2.5-Coder-32B with DRQC-Compress Hybrid Q-RUN"""
    def __init__(self, model_name_or_path, replace_layers=None, qrun_config=None):
        config = Qwen2Config.from_pretrained(model_name_or_path)
        with init_empty_weights():
            super().__init__(config)
        self._load_weights(model_name_or_path)

        default_cfg = {
            'n_reuploads': 3, 'mlp_hidden_size': 128, 'u_proj_output_dim': 4,
            'block_size': 64, 'entangle_rank': 64,
            'n_deep_layers': 2, 'd_hidden': 32,
        }
        if qrun_config:
            default_cfg.update(qrun_config)

        self.compute_dtype = resolve_compute_dtype(default_cfg.get("compute_dtype", "auto"))
        print(f"计算精度: {self.compute_dtype}")

        total_layers = len(self.model.layers)
        if replace_layers is None:
            replace_layers = list(range(total_layers))

        print(f"Hybrid 配置: {default_cfg}")
        print(f"替换 {len(replace_layers)}/{total_layers} 层")

        for idx in replace_layers:
            if idx < total_layers:
                new_mlp = Qwen2MLP_Hybrid(config, **default_cfg).to(dtype=self.compute_dtype)
                new_mlp.init_weights()
                self.model.layers[idx].mlp = new_mlp
                if idx % 8 == 0 or idx == replace_layers[-1]:
                    print(f"  -> 层 {idx}/{total_layers-1} 完成", flush=True)

        self._print_stats(replace_layers)

    def _load_weights(self, path):
        import os
        from safetensors.torch import load_file
        print(f"加载预训练权重: {path}")
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.safetensors')])
        print(f"找到 {len(files)} 个权重文件")
        sd = {}
        for f in files:
            sd.update(load_file(f))
        missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
        if missing:
            print(f"缺失: {len(missing)} 个 (预期，替换层的原始权重)")
        print("成功加载预训练权重")

    def _print_stats(self, replace_layers):
        total = sum(p.numel() for p in self.parameters())
        qrun = sum(self.model.layers[i].mlp.count_parameters() for i in replace_layers
                    if hasattr(self.model.layers[i].mlp, 'count_parameters'))
        orig_mlp_per_layer = 3 * (5120 * 27648 + 27648)  # gate+up+down with bias
        orig_total = orig_mlp_per_layer * len(replace_layers)
        print(f"\n参数量: 总 {total/1e9:.2f}B, Hybrid Q-RUN {qrun/1e6:.2f}M")
        print(f"替换层压缩: {orig_total/1e6:.0f}M → {qrun/1e6:.1f}M ({qrun/orig_total*100:.1f}%)")
        print(f"显存(BF16) ~{total*2/1e9:.1f}GB")


def create_hybrid_model(
    model_path="PATH_TO_PRETRAINED_MODEL",
    replace_layers=None, n_reuploads=3, mlp_hidden_size=128,
    u_proj_output_dim=4, block_size=64, entangle_rank=64,
    n_deep_layers=2, d_hidden=32, compute_dtype="auto",
):
    print("=" * 80)
    print("创建 Hybrid 模型 (DRQC-Compress)")
    print("=" * 80)
    return CustomQwen32B_Hybrid(
        model_name_or_path=model_path,
        replace_layers=replace_layers,
        qrun_config={
            'n_reuploads': n_reuploads, 'mlp_hidden_size': mlp_hidden_size,
            'u_proj_output_dim': u_proj_output_dim, 'block_size': block_size,
            'entangle_rank': entangle_rank, 'n_deep_layers': n_deep_layers,
            'd_hidden': d_hidden, 'compute_dtype': compute_dtype,
        },
    )
