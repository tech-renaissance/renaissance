"""Analyze BN vs no-BN experiment results and generate report."""
import os
import re
import glob

RESULTS_DIR = r"R:\renaissance\tests\bn\results"
REPORT_PATH = r"R:\renaissance\tests\bn\report1.md"

def parse_file(filepath):
    """Parse a result file and extract per-epoch validation accuracy."""
    accuracies = []
    best_epoch = None
    best_acc = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Match training log lines
            m = re.match(r'\s*\d+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)%', line)
            if m:
                accuracies.append(float(m.group(1)))
            # Match "Best Top-1: XX.XX%"
            m2 = re.match(r'\s*Best Top-1:\s*([\d.]+)%', line)
            if m2:
                best_acc = float(m2.group(1))
            # Match "Best Epoch: XX"
            m3 = re.match(r'\s*Best Epoch:\s*(\d+)', line)
            if m3:
                best_epoch = int(m3.group(1))
    return accuracies, best_acc, best_epoch

def compute_metrics(accuracies):
    """Compute key metrics from accuracy list."""
    if not accuracies:
        return None
    first_99 = None
    for i, acc in enumerate(accuracies):
        if acc >= 99.0:
            first_99 = i + 1
            break
    max_acc = max(accuracies)
    last10 = accuracies[-10:]
    last10_avg = sum(last10) / len(last10)
    return {
        'first_99_epoch': first_99,
        'max_acc': max_acc,
        'last10_avg': last10_avg,
    }

def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.txt')))

    results = {}  # (act, 'BN') -> metrics
    extras = {}   # (act, 'BN') -> (best_acc, best_epoch)

    for f in files:
        basename = os.path.basename(f)
        if '_bn_' in basename:
            group = 'BN'
            m = re.search(r'_bn_(\w+)\.txt$', basename)
        else:
            group = 'NoBN'
            m = re.search(r'_gpu_(\w+)\.txt$', basename)
        if not m:
            continue
        act = m.group(1)
        accs, best_acc, best_epoch = parse_file(f)
        metrics = compute_metrics(accs)
        results[(act, group)] = metrics
        extras[(act, group)] = (best_acc, best_epoch)

    # All activations found
    activations = sorted(set(k[0] for k in results.keys()))

    # Build report
    lines = []
    lines.append("# Batch Normalization (BN) 对比实验报告\n")
    lines.append("## 1. 实验设置\n")
    lines.append("- **网络结构**: 784→1024→(BN?)→512→(BN?)→256→(BN?)→10")
    lines.append("- **对比组**: 带BN vs 不带BN")
    lines.append("- **激活函数**: " + ", ".join(activations))
    lines.append("- **优化器**: AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)")
    lines.append("- **学习率调度**: CosineAnnealing + Warmup(5 epochs)")
    lines.append("- **数据增强**: Pad+Rotation+Scale+Crop+Autocontrast+Erasing")
    lines.append("- **训练**: 100 epochs, batch_size=128")
    lines.append("")
    lines.append("## 2. 对比指标\n")
    lines.append("| 指标 | 说明 |")
    lines.append("|------|------|")
    lines.append("| **首次 >=99.0% epoch** | 验证准确率首次达到 99.0% 的训练轮次，反映收敛速度 |")
    lines.append("| **最高准确率** | 100 个 epoch 中验证准确率的最高值，反映模型峰值性能 |")
    lines.append("| **最后10 epoch 均值** | 最后 10 个 epoch 验证准确率的平均值，反映模型最终稳定性 |")
    lines.append("")

    # Main comparison table
    lines.append("## 3. 详细数据对比\n")
    header = f"| 激活函数 | 组别 | 首次>=99.0% epoch | 最高准确率 | 最高epoch | 最后10 epoch 均值 |"
    lines.append(header)
    lines.append("|" + "|".join(["-" * w for w in [12, 8, 20, 14, 12, 20]]) + "|")

    for act in activations:
        for group in ['BN', 'NoBN']:
            key = (act, group)
            if key not in results:
                continue
            m = results[key]
            best_acc, best_epoch = extras[key]
            first99_str = str(m['first_99_epoch']) if m['first_99_epoch'] is not None else 'N/A'
            lines.append(f"| {act} | {group} | {first99_str} | {m['max_acc']:.2f}% | {best_epoch} | {m['last10_avg']:.2f}% |")

    lines.append("")

    # Difference table
    lines.append("## 4. BN 与 NoBN 差值对比 (BN - NoBN)\n")
    lines.append("| 激活函数 | Δ 首次>=99.0% epoch | Δ 最高准确率 | Δ 最后10 epoch 均值 |")
    lines.append("|----------|---------------------|-------------|---------------------|")

    sums = {'first99': 0, 'max': 0, 'last10': 0}
    count = 0
    for act in activations:
        bn = results.get((act, 'BN'))
        nobn = results.get((act, 'NoBN'))
        if bn and nobn:
            d1 = bn['first_99_epoch'] - nobn['first_99_epoch'] if bn['first_99_epoch'] and nobn['first_99_epoch'] else 0
            d2 = bn['max_acc'] - nobn['max_acc']
            d3 = bn['last10_avg'] - nobn['last10_avg']
            # Format with sign
            d1s = f"{d1:+d}" if d1 != 0 else "0"
            d2s = f"{d2:+.2f}%" if d2 != 0 else "0.00%"
            d3s = f"{d3:+.2f}%" if d3 != 0 else "0.00%"
            lines.append(f"| {act} | {d1s} | {d2s} | {d3s} |")
            sums['first99'] += d1
            sums['max'] += d2
            sums['last10'] += d3
            count += 1

    # Average row
    avg1 = sums['first99'] / count
    avg2 = sums['max'] / count
    avg3 = sums['last10'] / count
    lines.append(f"| **平均** | **{avg1:+.1f}** | **{avg2:+.2f}%** | **{avg3:+.2f}%** |")
    lines.append("")

    # Analysis
    lines.append("## 5. 分析解读\n")

    lines.append("### 5.1 收敛速度（首次 >=99.0% epoch）\n")
    lines.append("BN 层对大多数激活函数的收敛速度有明显的加速效果：\n")
    
    # Group activations by improvement direction
    improved = []
    worsened = []
    for act in activations:
        bn = results.get((act, 'BN'))
        nobn = results.get((act, 'NoBN'))
        if bn and nobn and bn['first_99_epoch'] and nobn['first_99_epoch']:
            d = bn['first_99_epoch'] - nobn['first_99_epoch']
            if d < 0:
                improved.append((act, d))
            elif d > 0:
                worsened.append((act, d))

    if improved:
        lines.append("- **收敛更快（BN 优势）**: " + ", ".join(
            f"{act}({d:+d} epochs)" for act, d in improved) + "\n")
    if worsened:
        lines.append("- **收敛更慢（BN 劣势）**: " + ", ".join(
            f"{act}({d:+d} epochs)" for act, d in worsened) + "\n")
    
    # Special case: tanh
    tn_bn = results.get(('tanh', 'BN'))
    tn_nobn = results.get(('tanh', 'NoBN'))
    if tn_bn and tn_nobn and tn_bn['first_99_epoch'] and tn_nobn['first_99_epoch']:
        d = tn_bn['first_99_epoch'] - tn_nobn['first_99_epoch']
        lines.append(f"\n特别值得注意的是 **tanh** 激活函数：不带 BN 时需 {tn_nobn['first_99_epoch']} 个 epoch 才首次达到 99.0%，而带 BN 仅需 {tn_bn['first_99_epoch']} 个 epoch，**提前了 {-d} 个 epoch**。这说明 BN 对 tanh 这类饱和激活函数的梯度消失问题有显著的缓解作用。\n")

    lines.append("### 5.2 最高准确率\n")
    lines.append("BN 对最终最高准确率的影响因激活函数而异，整体差异较小：\n")
    
    bn_better_max = []
    bn_worse_max = []
    for act in activations:
        bn = results.get((act, 'BN'))
        nobn = results.get((act, 'NoBN'))
        if bn and nobn:
            d = bn['max_acc'] - nobn['max_acc']
            if d > 0.005:
                bn_better_max.append((act, d))
            elif d < -0.005:
                bn_worse_max.append((act, d))

    if bn_better_max:
        lines.append("- **BN 略优**: " + ", ".join(f"{act}({d:+.2f}%)" for act, d in bn_better_max) + "\n")
    if bn_worse_max:
        lines.append("- **BN 略差**: " + ", ".join(f"{act}({d:+.2f}%)" for act, d in bn_worse_max) + "\n")
    
    lines.append("总体而言，最高准确率的差异在 0.1% 以内，可以认为 BN 对最终峰值性能没有显著影响。\n")

    lines.append("### 5.3 最终稳定性（最后10 epoch 均值）\n")
    lines.append("该指标反映训练后期模型的稳定程度：\n")
    
    for act in activations:
        bn = results.get((act, 'BN'))
        nobn = results.get((act, 'NoBN'))
        if bn and nobn:
            d = bn['last10_avg'] - nobn['last10_avg']
            if abs(d) > 0.01:
                better = "BN 更优" if d > 0 else "NoBN 更优"
                lines.append(f"- **{act}**: {better}，差值 {d:+.2f}%\n")
            else:
                lines.append(f"- **{act}**: 两者持平（差值 {d:+.2f}%）\n")

    lines.append("\n### 5.4 综合结论\n")
    lines.append("1. **收敛速度是 BN 的核心优势**：除 sigmoid 外，BN 使所有激活函数的收敛速度都有不同程度提升，" +
                 f"尤其在 tanh 上效果显著（提前 {abs(tn_bn['first_99_epoch'] - tn_nobn['first_99_epoch'])} 个 epoch）。\n")
    lines.append("2. **BN 对峰值准确率影响很小**：带 BN 与不带 BN 的最高准确率差异均小于 0.1%，可以认为 BN 不损害模型的峰值性能。\n")
    lines.append("3. **BN 对后期稳定性有轻微的负面影响**：最后 10 epoch 的均值，BN 组在大多数激活函数上略低于 NoBN 组。这可能是因为 BN 引入的随机性（batch statistics 的噪声）在训练后期导致轻微的波动。" +
                 " 不过差异极小（<0.1%），实际影响可以忽略。\n")
    lines.append("4. **sigmoid 是唯一例外**：带 BN 的 sigmoid 收敛速度反而慢于不带 BN 的版本。这可能是因为 sigmoid 本身的饱和特性与 BN 的标准化存在交互，导致梯度传播反而变差。\n")

    # Additional: best epoch comparison
    lines.append("## 6. 附录：最高准确率出现时机\n")
    lines.append("| 激活函数 | BN best epoch | NoBN best epoch | 差异 |")
    lines.append("|----------|--------------|----------------|------|")
    for act in activations:
        bn_best = extras.get((act, 'BN'), (None, None))
        nobn_best = extras.get((act, 'NoBN'), (None, None))
        if bn_best[1] and nobn_best[1]:
            d = bn_best[1] - nobn_best[1]
            lines.append(f"| {act} | {bn_best[1]} | {nobn_best[1]} | {d:+d} |")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report written to: {REPORT_PATH}")
    print(f"\n=== Report Preview ===\n")
    print(report)

if __name__ == '__main__':
    main()