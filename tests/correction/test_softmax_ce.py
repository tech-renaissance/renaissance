#!/usr/bin/env python3
"""
SOFTMAX_CE FWD+BWD 参考数据生成脚本（PyTorch）
用法：
  python test_softmax_ce.py --batch 8 --num_classes 1000 --seed 42 --workspace ./softmax_ce_data --dtype fp32
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

try:
    from tsr_v4 import save_tsr
except ImportError:
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python", "scripts")
    sys.path.insert(0, scripts_dir)
    from tsr_v4 import save_tsr

def generate_reference_data(batch, num_classes, seed, workspace, dtype_str):
    torch_dtype = torch.float32 if dtype_str == "fp32" else torch.float16

    os.makedirs(workspace, exist_ok=True)

    torch.manual_seed(seed)

    logits_nchw = torch.randn(batch, num_classes, 1, 1, dtype=torch_dtype) * 0.5
    labels = torch.randint(0, num_classes, (batch,), dtype=torch.long)

    scaling = 1.0
    inv_batch = 1.0 / batch

    # === FWD ===
    log_probs_nchw = F.log_softmax(logits_nchw.float(), dim=1)
    ce_loss_nchw = -log_probs_nchw[torch.arange(batch), labels].mean()
    softmax_probs_nchw = log_probs_nchw.exp()

    probs_2d = softmax_probs_nchw.squeeze(2).squeeze(2)  # [N, C]
    top1_probs, top1_indices = probs_2d.topk(1, dim=1)
    topk = min(5, num_classes)
    topk_probs, topk_indices = probs_2d.topk(topk, dim=1)

    top1_hits = (top1_indices.squeeze(1) == labels).sum().item()
    if num_classes >= 5:
        top5_hits = (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()
    else:
        top5_hits = batch

    top1_rate = float(top1_hits) / float(batch)
    top5_rate = float(top5_hits) / float(batch)

    pred_labels = top1_indices.squeeze(1).to(torch.int32)

    # === BWD ===
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    one_hot = one_hot.unsqueeze(2).unsqueeze(3)  # [N, C, 1, 1]
    d_logits_nchw = (softmax_probs_nchw - one_hot) * scaling * inv_batch

    # === NCHW -> NHWC ===
    logits_nhwc = logits_nchw.permute(0, 2, 3, 1).contiguous()
    softmax_probs_nhwc = softmax_probs_nchw.permute(0, 2, 3, 1).contiguous()
    d_logits_nhwc = d_logits_nchw.permute(0, 2, 3, 1).contiguous()

    labels_nhwc = labels.unsqueeze(1).unsqueeze(1).unsqueeze(1).contiguous().to(torch.int32)      # [N,1,1,1]
    pred_labels_nhwc = pred_labels.unsqueeze(1).unsqueeze(1).unsqueeze(1).contiguous()
    ce_loss_scalar = ce_loss_nchw.reshape(1,1,1,1)                          # [1,1,1,1]
    top1_scalar = torch.tensor([top1_rate], dtype=torch.float32).reshape(1,1,1,1)
    top5_scalar = torch.tensor([top5_rate], dtype=torch.float32).reshape(1,1,1,1)

    sfx = "_fp32"

    save_tsr(f"{workspace}/logits{sfx}.tsr",       [logits_nhwc.cpu().numpy()],         compress=False)
    save_tsr(f"{workspace}/labels{sfx}.tsr",       [labels_nhwc.cpu().numpy()],         compress=False)
    save_tsr(f"{workspace}/ce_loss{sfx}.tsr",      [ce_loss_scalar.cpu().numpy()],       compress=False)
    save_tsr(f"{workspace}/softmax_probs{sfx}.tsr",[softmax_probs_nhwc.cpu().numpy()],  compress=False)
    save_tsr(f"{workspace}/top1_correct{sfx}.tsr", [top1_scalar.cpu().numpy()],         compress=False)
    save_tsr(f"{workspace}/top5_correct{sfx}.tsr", [top5_scalar.cpu().numpy()],         compress=False)
    save_tsr(f"{workspace}/pred_labels{sfx}.tsr",  [pred_labels_nhwc.cpu().numpy()],    compress=False)
    save_tsr(f"{workspace}/d_logits{sfx}.tsr",     [d_logits_nhwc.cpu().numpy()],       compress=False)

    if dtype_str == "fp16":
        save_tsr(f"{workspace}/logits_amp.tsr",     [logits_nhwc.cpu().numpy()],         compress=False)
        save_tsr(f"{workspace}/d_logits_amp.tsr",   [d_logits_nhwc.to(torch.float16).cpu().numpy()], compress=False)
    else:
        save_tsr(f"{workspace}/logits_amp.tsr",     [logits_nhwc.cpu().numpy()],         compress=False)
        save_tsr(f"{workspace}/d_logits_amp.tsr",   [d_logits_nhwc.cpu().numpy()],       compress=False)

    print(f"Reference data generated successfully in {workspace}")
    print(f"  Batch: {batch}, Num Classes: {num_classes}, Seed: {seed}, Dtype: {dtype_str}")
    print(f"  CE Loss: {ce_loss_nchw.item():.6f}")
    print(f"  Top-1 Rate: {top1_rate:.6f} ({top1_hits}/{batch})")
    print(f"  Top-5 Rate: {top5_rate:.6f} ({top5_hits}/{batch})")

def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch reference data for SOFTMAX_CE")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace", type=str, default="./softmax_ce_data")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()

    generate_reference_data(args.batch, args.num_classes, args.seed, args.workspace, args.dtype)

if __name__ == "__main__":
    main()
