# 【小伙伴K】

• 当前诊断完成，汇报如下：
  已排除的问题：
  1. lr_dtensor_id_ 已正确修复并绑定，LR 通过 cudaMemcpyAsync 传入 device。
  2. ComputationGraph 结构正确：7 层 MLP，DEEP_FWD_BWD 包含完整 FWD+BWD。
  3. Tanh BWD 使用重计算（从 dx 位置读原始 x），不受 FC BWD in-place 覆盖影响。
  4. FC BWD 明确注释 Y_output unused，不依赖被覆盖的 Y。
  5. ZERO_GRAD 清除 R26+R27，覆盖全部 6 个梯度张量。
  6. Optimizer kernel（Momentum）公式与 PyTorch 一致。
  7. 学习率调度正确：epoch0=0.01, epoch1=0.0075。
  8. Scaling factor 初始化为 1.0，SoftmaxCE 内部 inv_scaling = 1/batch。
    尚未找到根因：
    • 所有组件的代码逻辑和图结构经过逐层检查，未发现明显的 bug。
    • 训练 epoch 时间仅 0.28s（469 batches），虽然对 MNIST MLP 在 RTX 5090 上可能正常，但结合 84% 的准确率，仍怀疑存在某种

    性的数值问题或流程遗漏。
  剩余怀疑方向（供参考）：
  1. FC BWD 的 cuBLAS 调用：dW 在 COMP_1，dX 在 COMP_3，跨流 event 同步在 CUDA Graph capture 中是否可靠。
  2. Weight initialization：TR 使用 KaimingUniform(gain=√2)，PyTorch nn.Linear 默认 KaimingUniform(a=√5)，方差差异可能导
     收敛变慢。
  3. 某些 batch 的梯度/loss 计算异常：最后一个 batch 只有 32 个样本但 kernel 按 128 处理，虽不应致命，但可能影响平均梯度
  4. CUDA Graph 运行时参数固化：如果 capture 时某些标量值被固化，运行时更新可能未生效（但 lr 通过 device memory 更新，应
     受此影响）。

  结论：诊断代码已添加，编译通过，运行输出完整。需要进一步缩小范围，建议在 run_train_epoch_gpu() 中添加逐 batch 的 loss/
  lr/梯度范数诊断，或对比 PyTorch 的逐层输出。