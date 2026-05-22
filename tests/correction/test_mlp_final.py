#!/usr/bin/env python3
"""
三层MLP (784-512-256-10) + SoftmaxCE 复合算子正确性测试 — PyTorch 参考数据生成
FWD: X[7,28,28,1] → Flatten → FC1 → Tanh → FC2 → Tanh → FC3 → SoftmaxCE
BWD: SoftmaxCE_BWD → FC3_BWD → Tanh2_BWD → FC2_BWD → Tanh1_BWD → FC1_BWD → Flatten_BWD
支持 --dtype fp16 | fp32
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'python', 'scripts'))
from tsr_v4 import save_tsr


def generate(seed: int, ws: str, dtype: str):
    if dtype == 'fp16':
        torch_dtype = torch.float16
        np_dtype    = np.float16
        suffix      = '_amp'
        dtype_label = 'FP16'
        is_fp16     = True
    else:
        torch_dtype = torch.float32
        np_dtype    = np.float32
        suffix      = '_fp32'
        dtype_label = 'FP32'
        is_fp16     = False

    if is_fp16:
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.backends.cudnn.fp32_precision = 'ieee'
        except AttributeError:
            pass

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(ws, exist_ok=True)

    # ── 固定形状 ──
    batch = 7
    H, W, C = 28, 28, 1
    flat_dim = H * W * C
    fc1_in, fc1_out_dim = flat_dim, 512
    fc2_in, fc2_out_dim = 512, 256
    fc3_in, fc3_out_dim = 256, 10
    num_classes = 10

    fc1 = torch.nn.Linear(fc1_in, fc1_out_dim, bias=True)
    fc2 = torch.nn.Linear(fc2_in, fc2_out_dim, bias=True)
    fc3 = torch.nn.Linear(fc3_in, fc3_out_dim, bias=True)
    fc1 = fc1.to(dtype=torch_dtype)
    fc2 = fc2.to(dtype=torch_dtype)
    fc3 = fc3.to(dtype=torch_dtype)

    with torch.no_grad():
        fc1.weight.copy_(torch.randn(fc1_out_dim, fc1_in, dtype=torch_dtype) * 0.1)
        fc1.bias.copy_(torch.randn(fc1_out_dim, dtype=torch.float32) * 0.1)
        fc2.weight.copy_(torch.randn(fc2_out_dim, fc2_in, dtype=torch_dtype) * 0.1)
        fc2.bias.copy_(torch.randn(fc2_out_dim, dtype=torch.float32) * 0.1)
        fc3.weight.copy_(torch.randn(fc3_out_dim, fc3_in, dtype=torch_dtype) * 0.1)
        fc3.bias.copy_(torch.randn(fc3_out_dim, dtype=torch.float32) * 0.1)

    # ── FWD ──
    X_nhwc = torch.randn(batch, H, W, C, dtype=torch_dtype) * 2.0
    labels = torch.randint(0, num_classes, (batch,), dtype=torch.int32)

    X_flat_2d = X_nhwc.reshape(batch, flat_dim)
    flat_out = X_flat_2d.view(batch, 1, 1, flat_dim)

    # FC1 FWD
    fc1_out_2d = fc1(X_flat_2d)
    fc1_out = fc1_out_2d.view(batch, 1, 1, fc1_out_dim)

    # Tanh1 FWD
    tanh1_out_2d = torch.tanh(fc1_out_2d)
    tanh1_out = tanh1_out_2d.view(batch, 1, 1, fc1_out_dim)

    # FC2 FWD
    fc2_out_2d = fc2(tanh1_out_2d)
    fc2_out = fc2_out_2d.view(batch, 1, 1, fc2_out_dim)

    # Tanh2 FWD
    tanh2_out_2d = torch.tanh(fc2_out_2d)
    tanh2_out = tanh2_out_2d.view(batch, 1, 1, fc2_out_dim)

    # FC3 FWD
    logits_2d = fc3(tanh2_out_2d)
    logits = logits_2d.view(batch, 1, 1, fc3_out_dim)

    # Softmax + CE
    probs_2d = F.softmax(logits_2d, dim=1)
    ce_loss = F.cross_entropy(logits_2d, labels.to(torch.long))
    inv_scaling = 1.0 / batch
    scaling = 1.0

    # ── BWD (手动链式法则，匹配项目 in-place 语义) ──
    # SoftmaxCE BWD: d_logits = (probs - one_hot(labels)) / batch
    d_logits_2d = probs_2d.clone()
    d_logits_2d[range(batch), labels] -= 1.0
    d_logits_2d /= batch
    d_logits = d_logits_2d.view(batch, 1, 1, fc3_out_dim)

    # FC3 BWD
    W3 = fc3.weight.data
    d_tanh2_2d = torch.mm(d_logits_2d, W3)
    d_tanh2 = d_tanh2_2d.view(batch, 1, 1, fc2_out_dim)
    dW3_2d = torch.mm(d_logits_2d.t(), tanh2_out_2d)
    dW3 = dW3_2d.view(fc3_out_dim, 1, 1, fc3_in).contiguous()
    dB3_2d = d_logits_2d.sum(dim=0).to(torch.float32)
    dB3 = dB3_2d.view(1, 1, 1, fc3_out_dim).contiguous()

    # Tanh2 BWD: dx = dy * (1 - tanh(x)^2) = dy * (1 - tanh2_out^2)
    # 注意：tanh2_out = tanh(fc2_out)，所以导数是 (1 - tanh2_out^2)
    d_fc2_2d = d_tanh2_2d * (1.0 - tanh2_out_2d ** 2)
    d_fc2 = d_fc2_2d.view(batch, 1, 1, fc2_out_dim)

    # FC2 BWD
    W2 = fc2.weight.data
    d_tanh1_2d = torch.mm(d_fc2_2d, W2)
    d_tanh1 = d_tanh1_2d.view(batch, 1, 1, fc1_out_dim)
    dW2_2d = torch.mm(d_fc2_2d.t(), tanh1_out_2d)
    dW2 = dW2_2d.view(fc2_out_dim, 1, 1, fc1_out_dim).contiguous()
    dB2_2d = d_fc2_2d.sum(dim=0).to(torch.float32)
    dB2 = dB2_2d.view(1, 1, 1, fc2_out_dim).contiguous()

    # Tanh1 BWD
    d_fc1_2d = d_tanh1_2d * (1.0 - tanh1_out_2d ** 2)
    d_fc1 = d_fc1_2d.view(batch, 1, 1, fc1_out_dim)

    # FC1 BWD
    W1 = fc1.weight.data
    d_flat_2d = torch.mm(d_fc1_2d, W1)
    d_flat = d_flat_2d.view(batch, 1, 1, flat_dim)
    dW1_2d = torch.mm(d_fc1_2d.t(), X_flat_2d)
    dW1 = dW1_2d.view(fc1_out_dim, 1, 1, flat_dim).contiguous()
    dB1_2d = d_fc1_2d.sum(dim=0).to(torch.float32)
    dB1 = dB1_2d.view(1, 1, 1, fc1_out_dim).contiguous()

    # Flatten BWD
    d_x = d_flat_2d.reshape(batch, H, W, C).contiguous()

    # ── Weight NHWC ──
    W1_4d = W1.view(fc1_out_dim, 1, 1, flat_dim).contiguous()
    W2_4d = W2.view(fc2_out_dim, 1, 1, fc1_out_dim).contiguous()
    W3_4d = W3.view(fc3_out_dim, 1, 1, fc3_in).contiguous()
    B1_4d = fc1.bias.data.to(torch.float32).view(1, 1, 1, fc1_out_dim).contiguous()
    B2_4d = fc2.bias.data.to(torch.float32).view(1, 1, 1, fc2_out_dim).contiguous()
    B3_4d = fc3.bias.data.to(torch.float32).view(1, 1, 1, fc3_out_dim).contiguous()

    # ── 保存 ──
    def s(name, data, np_dtype_override=None):
        dt = np_dtype_override if np_dtype_override else np_dtype
        save_tsr(os.path.join(ws, f'{name}{suffix}.tsr'),
                 [data.detach().cpu().numpy().astype(dt)], compress=False)

    # FWD
    s('x',         X_nhwc)
    s('flat_out',  flat_out)
    s('w1',        W1_4d)
    s('b1',        B1_4d, np.float32)
    s('fc1_out',   fc1_out)
    s('tanh1_out', tanh1_out)
    s('w2',        W2_4d)
    s('b2',        B2_4d, np.float32)
    s('fc2_out',   fc2_out)
    s('tanh2_out', tanh2_out)
    s('w3',        W3_4d)
    s('b3',        B3_4d, np.float32)
    s('logits',    logits)
    s('probs',     probs_2d.view(batch, 1, 1, num_classes), np.float32)
    s('loss',      torch.tensor([ce_loss.item()], dtype=torch.float32).view(1,1,1,1), np.float32)
    s('inv_scaling', torch.tensor([inv_scaling], dtype=torch.float32).view(1,1,1,1), np.float32)
    s('labels',    labels.view(batch, 1, 1, 1).to(torch.int32), np.int32)

    # BWD refs
    s('d_logits',  d_logits)
    s('dw3_ref',   dW3)
    s('db3_ref',   dB3, np.float32)
    s('d_tanh2_ref', d_tanh2)
    s('d_fc2_ref', d_fc2)
    s('dw2_ref',   dW2)
    s('db2_ref',   dB2, np.float32)
    s('d_tanh1_ref', d_tanh1)
    s('d_fc1_ref', d_fc1)
    s('dw1_ref',   dW1)
    s('db1_ref',   dB1, np.float32)
    s('d_flat_ref', d_flat)
    s('d_x_ref',   d_x)

    file_desc = (
        f"  x:         x{suffix}.tsr\n"
        f"  flat_out:  flat_out{suffix}.tsr\n"
        f"  w1:        w1{suffix}.tsr\n"
        f"  b1:        b1{suffix}.tsr\n"
        f"  fc1_out:   fc1_out{suffix}.tsr\n"
        f"  tanh1_out: tanh1_out{suffix}.tsr\n"
        f"  w2:        w2{suffix}.tsr\n"
        f"  b2:        b2{suffix}.tsr\n"
        f"  fc2_out:   fc2_out{suffix}.tsr\n"
        f"  tanh2_out: tanh2_out{suffix}.tsr\n"
        f"  w3:        w3{suffix}.tsr\n"
        f"  b3:        b3{suffix}.tsr\n"
        f"  logits:    logits{suffix}.tsr\n"
        f"  probs:     probs{suffix}.tsr\n"
        f"  loss:      loss{suffix}.tsr\n"
        f"  d_logits:  d_logits{suffix}.tsr\n"
        f"  dw3_ref:   dw3_ref{suffix}.tsr\n"
        f"  db3_ref:   db3_ref{suffix}.tsr\n"
        f"  d_tanh2_ref: d_tanh2_ref{suffix}.tsr\n"
        f"  d_fc2_ref: d_fc2_ref{suffix}.tsr\n"
        f"  dw2_ref:   dw2_ref{suffix}.tsr\n"
        f"  db2_ref:   db2_ref{suffix}.tsr\n"
        f"  d_tanh1_ref: d_tanh1_ref{suffix}.tsr\n"
        f"  d_fc1_ref: d_fc1_ref{suffix}.tsr\n"
        f"  dw1_ref:   dw1_ref{suffix}.tsr\n"
        f"  db1_ref:   db1_ref{suffix}.tsr\n"
        f"  d_flat_ref: d_flat_ref{suffix}.tsr\n"
        f"  d_x_ref:   d_x_ref{suffix}.tsr\n"
    )

    yaml = (
        f"op: mlp_final\n"
        f"dtype: {dtype_label}\n"
        f"shape: [{batch}, {H}, {W}, {C}] (NHWC)\n"
        f"fc1: {fc1_in}→{fc1_out_dim}\n"
        f"fc2: {fc2_in}→{fc2_out_dim}\n"
        f"fc3: {fc3_in}→{fc3_out_dim}\n"
        f"seed: {seed}\n"
        f"mse_threshold: {'1e-3' if is_fp16 else '1e-5'}\n"
        f"files:\n{file_desc}"
    )
    with open(os.path.join(ws, 'config.yaml'), 'w') as f:
        f.write(yaml)

    print(f"MLP Final {dtype_label} reference generated in: {ws}")


def main():
    parser = argparse.ArgumentParser(
        description='MLP Final (Flatten+FC+Tanh+FC+Tanh+FC+SoftmaxCE) test data generator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--dtype', default='fp32', choices=['fp16', 'fp32'])
    args = parser.parse_args()
    generate(args.seed, args.workspace, args.dtype)


if __name__ == '__main__':
    main()
