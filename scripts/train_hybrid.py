"""
train_hybrid.py — DRQC-Compress 混合架构训练脚本

导入 CustomQwen32B_hybrid 中的 create_hybrid_model。
"""
import argparse
import json
import csv
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import dispatch_model, infer_auto_device_map
from CustomQwen32B_hybrid import create_hybrid_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"


def resolve_runtime_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def get_device_count(device):
    if device == "cuda":
        return torch.cuda.device_count()
    if device == "musa" and hasattr(torch, "musa"):
        return torch.musa.device_count()
    if device == "npu" and hasattr(torch, "npu"):
        return torch.npu.device_count()
    return 1


def get_input_device(model):
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(text, max_length=self.max_length, truncation=True,
                                padding="max_length", return_tensors="pt")
        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False,
                                                          add_generation_prompt=True)
        prompt_len = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        labels = tokens["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": labels.squeeze()}


def main():
    parser = argparse.ArgumentParser(description="Hybrid DRQC-Compress 训练")
    parser.add_argument("--model_path", type=str, default="/private/models/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--data_path", type=str, default="/private/THeWakeSystems-QRUN-Qwen2.5-coder-32B/large_distill_data.json")
    parser.add_argument("--replace_layers", type=int, nargs="+", default=list(range(48, 64)))
    parser.add_argument("--u_proj_output_dim", type=int, default=4)
    parser.add_argument("--mlp_hidden_size", type=int, default=128)
    parser.add_argument("--n_reuploads", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--entangle_rank", type=int, default=64)
    parser.add_argument("--n_deep_layers", type=int, default=2)
    parser.add_argument("--d_hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="/private/THeWakeSystems-QRUN-Qwen2.5-coder-32B/checkpoints_hybrid_v2")
    parser.add_argument("--save_every_n_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_memory_per_gpu", type=str, default="18GiB")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "musa", "npu", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--resume", type=str, default=None, help="断点续训: checkpoint 路径")
    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid DRQC-Compress 训练")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_hybrid_model(
        model_path=args.model_path,
        replace_layers=args.replace_layers,
        n_reuploads=args.n_reuploads,
        mlp_hidden_size=args.mlp_hidden_size,
        u_proj_output_dim=args.u_proj_output_dim,
        block_size=args.block_size,
        entangle_rank=args.entangle_rank,
        n_deep_layers=args.n_deep_layers,
        d_hidden=args.d_hidden,
        compute_dtype=args.dtype,
    )

    # 冻结非替换层
    replaced_set = set(args.replace_layers)
    for name, param in model.named_parameters():
        is_replaced = any(f"layers.{i}." in name for i in replaced_set)
        param.requires_grad = is_replaced and "mlp" in name

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"可训练: {trainable_count/1e6:.2f}M / {total_count/1e9:.2f}B")

    model.gradient_checkpointing_enable()

    runtime_device = resolve_runtime_device(args.device)
    n_devices = get_device_count(runtime_device)
    if runtime_device == "cuda" and n_devices > 1:
        print(f"\n{n_devices} 张 CUDA GPU，自动分配...")
        max_mem = {i: args.max_memory_per_gpu for i in range(n_devices)}
        max_mem["cpu"] = "200GiB"
        dmap = infer_auto_device_map(model, max_memory=max_mem,
                                      no_split_module_classes=["Qwen2DecoderLayer"])
        model = dispatch_model(model, device_map=dmap)
        from collections import Counter
        gpu_counts = Counter(v for v in dmap.values() if isinstance(v, int))
        for dev, cnt in sorted(gpu_counts.items()):
            print(f"  {dev}: {cnt} 模块")
        if "cpu" in dmap.values():
            cpu_cnt = sum(1 for v in dmap.values() if v == "cpu")
            print(f"  cpu: {cpu_cnt} 模块")
    elif runtime_device != "cpu":
        target = f"{runtime_device}:0"
        model = model.to(target)
        print(f"\n使用设备: {target}")
    else:
        print("\n使用设备: cpu")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数张量: {len(trainable_params)}")

    dataset = SFTDataset(args.data_path, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"数据: {len(dataset)} 条")

    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 断点续训
    start_epoch = 0
    start_step = 0
    global_step = 0
    if args.resume:
        print(f"\n从断点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        # 恢复模型权重
        for n, p in model.named_parameters():
            if n in ckpt["model_state_dict"]:
                p.data.copy_(ckpt["model_state_dict"][n].to(p.device))
        # 恢复 optimizer 和 scheduler
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"  恢复到 epoch={start_epoch}, step={start_step}, global_step={global_step}")
        del ckpt

    os.makedirs(args.save_path, exist_ok=True)
    log_f = open(os.path.join(args.save_path, "training_log.csv"), "a" if args.resume else "w", newline="")
    writer = csv.writer(log_f)
    if not args.resume:
        writer.writerow(["epoch", "step", "global_step", "loss", "avg_loss", "lr"])

    print("\n开始训练...")
    model.train()
    nan_count = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        valid_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            # 跳过已完成的 step（断点续训）
            if epoch == start_epoch and step < start_step:
                continue
            input_device = get_input_device(model)
            ids = batch["input_ids"].to(input_device, non_blocking=True)
            mask = batch["attention_mask"].to(input_device, non_blocking=True)
            labs = batch["labels"].to(input_device, non_blocking=True)

            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            loss = out.loss / args.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                if nan_count > 50:
                    print("[ERROR] NaN 过多，终止")
                    log_f.close()
                    return
                continue

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                    sf = os.path.join(args.save_path, f"step_{global_step}.pt")
                    sd = {n: p.cpu() for n, p in model.named_parameters() if p.requires_grad}
                    torch.save({"epoch": epoch, "step": step + 1, "global_step": global_step,
                                "model_state_dict": sd,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "args": vars(args)}, sf)
                    print(f"  [CKPT] {sf}")

            actual = loss.item() * args.gradient_accumulation_steps
            total_loss += actual
            valid_steps += 1

            if step % 100 == 0:
                avg = total_loss / max(valid_steps, 1)
                lr = scheduler.get_last_lr()[0]
                spd = (epoch * len(dataloader) + step + 1) / (time.time() - t0)
                print(f"  E{epoch+1} S{step}/{len(dataloader)} L:{actual:.4f} Avg:{avg:.4f} LR:{lr:.2e} {spd:.1f}s/s")

            writer.writerow([epoch+1, step, global_step, f"{actual:.6f}",
                             f"{total_loss/max(valid_steps,1):.6f}", f"{scheduler.get_last_lr()[0]:.8f}"])

        avg = total_loss / max(valid_steps, 1)
        print(f"Epoch {epoch+1} 完成, Avg Loss: {avg:.4f}, NaN: {nan_count}")

        sf = os.path.join(args.save_path, f"epoch_{epoch+1}.pt")
        sd = {n: p.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"epoch": epoch + 1, "step": 0, "global_step": global_step,
                     "model_state_dict": sd,
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(),
                     "args": vars(args)}, sf)
        print(f"  已保存: {sf}")

    log_f.close()
    print(f"\n训练完成！")


if __name__ == "__main__":
    main()
