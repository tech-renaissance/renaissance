import os, glob, re

def parse_file(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|', line)
            if m:
                epoch = int(m.group(1))
                acc = float(m.group(4))
                data.append({'epoch': epoch, 'acc': acc})
    return data

def analyze(data):
    first_99 = None
    for d in data:
        if d['acc'] >= 99.0:
            first_99 = d['epoch']
            break
    max_acc = max(d['acc'] for d in data) if data else 0.0
    last10 = data[-10:] if len(data) >= 10 else data
    avg_last10 = sum(d['acc'] for d in last10) / len(last10) if last10 else 0.0
    return {'first_99': first_99, 'max_acc': max_acc, 'avg_last10': avg_last10}

results_dir = r'R:\renaissance\tests\bn\results'
files = glob.glob(os.path.join(results_dir, '*.txt'))

activations = {}
for f in files:
    name = os.path.basename(f)
    m = re.match(r'mnist_best_adamw_gpu_(?:bn_)?(.+)\.txt$', name)
    if m:
        act = m.group(1)
        has_bn = '_bn_' in name
        if act not in activations:
            activations[act] = {}
        activations[act][has_bn] = analyze(parse_file(f))

report_path = r'R:\renaissance\tests\bn\report2.md'

lines = []
lines.append('# MNIST 有 BN vs 无 BN 对比报告')
lines.append('')
lines.append('> 数据来源：`tests/bn/results/` 目录下 14 个 txt 文件')
lines.append('> - 带 `_bn` 后缀：有 BatchNorm (`mnist_best_adamw_bn`)')
lines.append('> - 不带 `_bn` 后缀：无 BatchNorm (`mnist_best_adamw`)')
lines.append('> - 每种激活函数各运行 100 epochs')
lines.append('')
lines.append('## 分析维度')
lines.append('')
lines.append('1. **首次达到 99.0% 准确率的 epoch**：衡量收敛速度')
lines.append('2. **最高准确率**：衡量模型上限')
lines.append('3. **最后 10 个 epoch 的平均准确率**：衡量最终收敛稳定性')
lines.append('')
lines.append('> 注：本报告不对比 train loss / val loss（PyTorch 使用 BN 后 loss 反而更高，属已知现象），也不对比用时（未控制后台进程，不准确）。')
lines.append('')
lines.append('## 汇总表')
lines.append('')
lines.append('| 激活函数 | 指标 | 无 BN | 有 BN | 差值 |')
lines.append('|---|---|---|---|---|')

for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        f99_nb = str(no_bn['first_99']) if no_bn['first_99'] else '未达'
        f99_b = str(bn['first_99']) if bn['first_99'] else '未达'
        delta_f99 = ''
        if no_bn['first_99'] and bn['first_99']:
            d = bn['first_99'] - no_bn['first_99']
            delta_f99 = f'{d:+d}'
        elif no_bn['first_99'] and not bn['first_99']:
            delta_f99 = 'N/A'
        elif not no_bn['first_99'] and bn['first_99']:
            delta_f99 = 'N/A'
        
        lines.append(f'| {act} | 首次 99.0% epoch | {f99_nb} | {f99_b} | {delta_f99} |')
        
        dmax = bn['max_acc'] - no_bn['max_acc']
        lines.append(f"| {act} | 最高准确率 | {no_bn['max_acc']:.2f}% | {bn['max_acc']:.2f}% | {dmax:+.2f}% |")
        
        davg = bn['avg_last10'] - no_bn['avg_last10']
        lines.append(f"| {act} | 最后10轮均值 | {no_bn['avg_last10']:.2f}% | {bn['avg_last10']:.2f}% | {davg:+.2f}% |")

lines.append('')

# Detailed analysis
lines.append('## 分项解读')
lines.append('')

# 1. 收敛速度
lines.append('### 1. 首次达到 99.0% 准确率的 epoch（越小越好）')
lines.append('')
lines.append('| 激活函数 | 无 BN | 有 BN | 对比 |')
lines.append('|---|---|---|---|')
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        f99_nb = no_bn['first_99'] if no_bn['first_99'] else None
        f99_b = bn['first_99'] if bn['first_99'] else None
        if f99_nb and f99_b:
            delta = f99_nb - f99_b
            if delta > 0:
                note = f'BN 快 {delta} 轮'
            elif delta < 0:
                note = f'BN 慢 {abs(delta)} 轮'
            else:
                note = '持平'
            lines.append(f'| {act} | {f99_nb} | {f99_b} | {note} |')
        elif f99_nb and not f99_b:
            lines.append(f'| {act} | {f99_nb} | 未达 | 无 BN 更快 |')
        elif not f99_nb and f99_b:
            lines.append(f'| {act} | 未达 | {f99_b} | BN 更快 |')
        else:
            lines.append(f'| {act} | 未达 | 未达 | 持平 |')

lines.append('')
lines.append('**观察：**')
lines.append('')

speedups = []
slowdowns = []
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn and no_bn['first_99'] and bn['first_99']:
        delta = no_bn['first_99'] - bn['first_99']
        if delta > 0:
            speedups.append((act, delta))
        elif delta < 0:
            slowdowns.append((act, -delta))

if speedups:
    su = '，'.join([f'{a}(+{d})' for a, d in speedups])
    lines.append(f'- BN 加速收敛：{su}')
if slowdowns:
    sl = '，'.join([f'{a}(-{d})' for a, d in slowdowns])
    lines.append(f'- BN 拖慢收敛：{sl}')

lines.append('')

# 2. 最高准确率
lines.append('### 2. 最高准确率（越高越好）')
lines.append('')
lines.append('| 激活函数 | 无 BN | 有 BN | 差值 |')
lines.append('|---|---|---|---|')
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        d = bn['max_acc'] - no_bn['max_acc']
        lines.append(f"| {act} | {no_bn['max_acc']:.2f}% | {bn['max_acc']:.2f}% | {d:+.2f}% |")

lines.append('')
lines.append('**观察：**')
lines.append('')

improved = []
declined = []
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        d = bn['max_acc'] - no_bn['max_acc']
        if d > 0:
            improved.append((act, d))
        elif d < 0:
            declined.append((act, d))

if improved:
    lines.append(f'- BN 提升上限：' + '，'.join([f'{a}(+{d:.2f}%)' for a, d in improved]))
if declined:
    lines.append(f'- BN 降低上限：' + '，'.join([f'{a}({d:.2f}%)' for a, d in declined]))

lines.append('')

# 3. 最后10轮均值
lines.append('### 3. 最后 10 个 epoch 的平均准确率（越高越好）')
lines.append('')
lines.append('| 激活函数 | 无 BN | 有 BN | 差值 |')
lines.append('|---|---|---|---|')
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        d = bn['avg_last10'] - no_bn['avg_last10']
        lines.append(f"| {act} | {no_bn['avg_last10']:.2f}% | {bn['avg_last10']:.2f}% | {d:+.2f}% |")

lines.append('')
lines.append('**观察：**')
lines.append('')

improved10 = []
declined10 = []
flat10 = []
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        d = bn['avg_last10'] - no_bn['avg_last10']
        if abs(d) < 0.01:
            flat10.append(act)
        elif d > 0:
            improved10.append((act, d))
        else:
            declined10.append((act, d))

if improved10:
    lines.append(f'- BN 提升稳定性：' + '，'.join([f'{a}(+{d:.2f}%)' for a, d in improved10]))
if flat10:
    lines.append(f'- 基本持平：' + '，'.join(flat10))
if declined10:
    lines.append(f'- BN 降低稳定性：' + '，'.join([f'{a}({d:.2f}%)' for a, d in declined10]))

lines.append('')

# Overall conclusion
lines.append('## 总体结论')
lines.append('')

speed_wins = sum(1 for act in activations if activations[act].get(False) and activations[act].get(True) 
                 and activations[act][False]['first_99'] and activations[act][True]['first_99']
                 and activations[act][False]['first_99'] > activations[act][True]['first_99'])
speed_losses = sum(1 for act in activations if activations[act].get(False) and activations[act].get(True) 
                   and activations[act][False]['first_99'] and activations[act][True]['first_99']
                   and activations[act][False]['first_99'] < activations[act][True]['first_99'])

max_wins = sum(1 for act in activations if activations[act].get(False) and activations[act].get(True)
               and activations[act][True]['max_acc'] > activations[act][False]['max_acc'])
max_losses = sum(1 for act in activations if activations[act].get(False) and activations[act].get(True)
                 and activations[act][True]['max_acc'] < activations[act][False]['max_acc'])

avg_wins = sum(1 for act in activations if activations[act].get(False) and activations[act].get(True)
               and activations[act][True]['avg_last10'] > activations[act][False]['avg_last10'] + 0.01)
avg_losses = sum(1 for act in activations if activations[act].get(False) and activations[act].get(True)
                 and activations[act][True]['avg_last10'] < activations[act][False]['avg_last10'] - 0.01)

lines.append(f'- **收敛速度**：{speed_wins} 种激活函数 BN 更快，{speed_losses} 种更慢（共 {len(activations)} 种）')
lines.append(f'- **最高准确率**：{max_wins} 种激活函数 BN 更高，{max_losses} 种更低')
lines.append(f'- **最终稳定性**：{avg_wins} 种激活函数 BN 更好，{avg_losses} 种更差')
lines.append('')

# Key findings
lines.append('### 关键发现')
lines.append('')

best_speed = None
best_max = None
best_avg = None
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        if no_bn['first_99'] and bn['first_99']:
            d = no_bn['first_99'] - bn['first_99']
            if best_speed is None or d > best_speed[1]:
                best_speed = (act, d)
        dmax = bn['max_acc'] - no_bn['max_acc']
        if best_max is None or dmax > best_max[1]:
            best_max = (act, dmax)
        davg = bn['avg_last10'] - no_bn['avg_last10']
        if best_avg is None or davg > best_avg[1]:
            best_avg = (act, davg)

if best_speed and best_speed[1] > 0:
    lines.append(f"1. **BN 对收敛速度帮助最大的是 `{best_speed[0]}`**：从 {activations[best_speed[0]][False]['first_99']} 轮提前到 {activations[best_speed[0]][True]['first_99']} 轮，节省了 {best_speed[1]} 个 epoch。")
if best_max and best_max[1] > 0:
    lines.append(f"2. **BN 对准确率上限提升最大的是 `{best_max[0]}`**：最高准确率提升 {best_max[1]:.2f}%。")
if best_avg and best_avg[1] > 0:
    lines.append(f"3. **BN 对最终稳定性提升最大的是 `{best_avg[0]}`**：最后 10 轮均值提升 {best_avg[1]:.2f}%。")

lines.append('')

worst_speed = None
worst_max = None
worst_avg = None
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        if no_bn['first_99'] and bn['first_99']:
            d = no_bn['first_99'] - bn['first_99']
            if worst_speed is None or d < worst_speed[1]:
                worst_speed = (act, d)
        dmax = bn['max_acc'] - no_bn['max_acc']
        if worst_max is None or dmax < worst_max[1]:
            worst_max = (act, dmax)
        davg = bn['avg_last10'] - no_bn['avg_last10']
        if worst_avg is None or davg < worst_avg[1]:
            worst_avg = (act, davg)

if worst_speed and worst_speed[1] < 0:
    lines.append(f"4. **BN 拖慢收敛的例外：`{worst_speed[0]}`**：从 {activations[worst_speed[0]][False]['first_99']} 轮推迟到 {activations[worst_speed[0]][True]['first_99']} 轮，慢了 {abs(worst_speed[1])} 个 epoch。")
if worst_max and worst_max[1] < 0:
    lines.append(f"5. **BN 降低上限的主要是 `{worst_max[0]}`**：最高准确率下降 {abs(worst_max[1]):.2f}%。")

lines.append('')
lines.append('### 综合判断')
lines.append('')

bn_beneficial = []
bn_harmful = []
for act in sorted(activations.keys()):
    no_bn = activations[act].get(False)
    bn = activations[act].get(True)
    if no_bn and bn:
        score = 0
        if no_bn['first_99'] and bn['first_99'] and bn['first_99'] < no_bn['first_99']:
            score += 1
        if bn['max_acc'] > no_bn['max_acc']:
            score += 1
        if bn['avg_last10'] > no_bn['avg_last10'] + 0.01:
            score += 1
        if score >= 2:
            bn_beneficial.append(act)
        elif score <= 0:
            bn_harmful.append(act)

if bn_beneficial:
    lines.append(f'- **建议使用 BN 的激活函数**：{", ".join(bn_beneficial)}')
if bn_harmful:
    lines.append(f'- **建议不使用 BN 的激活函数**：{", ".join(bn_harmful)}')

lines.append('')
lines.append('---')
lines.append('')
lines.append('*报告生成时间：2026-06-08*')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'Report written to {report_path}')
