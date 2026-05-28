# Batch 0 Tensor 对比分析 (GPU FP32 vs AMP FP16)

## 测试配置
- 随机种子: 42
- Batch size: 200
- 模型: MLP 784-512-256-10 (Tanh激活)
- 优化器: SGD(momentum=0.9)
- 学习率: 0.1

## 关键差异发现

### 1. 输入数据格式差异
| Tensor | GPU FP32 | AMP FP16 |
|--------|----------|----------|
| I_A_DATA id=1 | shape=[200,28,28,1] dtype=FP32 | shape=[200,28,28,4] dtype=FP16 |
| I_B_DATA id=3 | shape=[200,28,28,1] dtype=FP32 | shape=[200,28,28,4] dtype=FP16 |

### 2. Scaling Factor 差异
| Tensor | GPU FP32 | AMP FP16 |
|--------|----------|----------|
| S_SCALAR_FP32 id=7 (scaling) | **1.0** | **65536.0** |

### 3. Loss 值对比 (正常范围)
| Tensor | GPU FP32 | AMP FP16 | 差异 |
|--------|----------|----------|------|
| R_RESULT id=8 (loss) | **2.33601** | **2.30607** | ~1.3% (正常FP16精度损失) |

### 4. ⚠️ 关键问题：梯度值差异
| Tensor | GPU FP32 | AMP FP16 | 倍数差异 |
|--------|----------|----------|----------|
| G_FC_WEIGHT id=23 (第1层权重梯度) | **0.000662669** | **17.2702** | **~26047倍** |
| G_FC_BIAS id=24 (第1层偏置梯度) | **-0.00156211** | **-40.7012** | **~26061倍** |
| G_FC_WEIGHT id=33 (第2层权重梯度) | **-0.00151174** | **-52.6332** | **~34798倍** |
| G_FC_WEIGHT id=43 (第3层权重梯度) | **-0.011354** | **-453.479** | **~39934倍** |

### 5. 🔥 致命问题：FP16梯度全零
| Tensor | GPU FP32 | AMP FP16 |
|--------|----------|----------|
| G_FC_WEIGHT_FP16 id=26 | **N/A** | **0 0 0 0 0 0 0 0** (全零!) |
| G_FC_WEIGHT_FP16 id=36 | **N/A** | **0 0 0 0 0 0 0 0** (全零!) |
| G_FC_WEIGHT_FP16 id=46 | **N/A** | **0 0 0 0 0 0 0 0** (全零!) |

## 问题诊断

### P0-B Bug 确认：优化器缺少 unscaling
梯度被放大了 **scaling = 65536** 倍，但在Momentum优化器中没有除以scaling恢复：

```cpp
// 当前错误代码 (momentum_update_cpu):
m[i] = m[i] * beta + g[i];  // g[i]已被放大65536倍
w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];  // 使用了放大的梯度
```

应该是：
```cpp
float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
float g_i = g[i] * inv_scaling;  // 先unscaling
m[i] = m[i] * beta + g_i;        // 使用恢复后的梯度
w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
```

### FP16梯度全零问题
AMP版本中的FP16梯度tensor (`G_FC_WEIGHT_FP16`) 全部为0，这可能导致：
1. 梯度检查机制失效
2. 无法在FP16精度下进行梯度裁剪
3. 可能影响混合精度训练的数值稳定性

## 验证计划
根据 NQB_FINAL.md，需要修复：
1. **P0-A**: compiler.cpp optimizer input_ids 顺序错误
2. **P0-B**: 为所有非SGD优化器添加 unscaling (当前确认的主要问题)
3. **P0-C**: g_accum 执行顺序错误
4. **P1-0**: FWD loss 错误乘 scaling
5. **P1-1**: INF loss 错误乘 scaling

## 结论
当前的AMP训练失败主要由 **P0-B (优化器缺少unscaling)** 导致，梯度被放大65536倍后直接用于权重更新，完全破坏了训练过程。这与NQB_FINAL.md的分析完全吻合。