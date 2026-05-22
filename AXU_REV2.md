# 【小伙伴K】

• 对照代码逐项复核，AXU_FINAL.md 修复后整体质量很高，我提出的4条意见均已准确采纳。但仍发现 2 处残余问题：
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 1：第二步表格与正文不一致
  位置：§三、第二步 表格 L213-214
  问题：表格中 CPU launcher 仍写：
  ▌ 改为遍历 op_ctx->output_ranges + mp->resolve_region_bounds()
  但正文 L106-113 已明确说明 capture_cpu.cpp 已在调用前完成解析，CPU launcher 无需再调用 resolve_region_bounds()。
  建议：表格中 CPU 行改为：
  ▌ CPU launcher：删 L144 get_range_op_range()，改为遍历 op_ctx->output_ranges（offset/size 已由 capture_cpu.cpp 预解
  ▌ 析）
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  问题 2：P3 隐患范围描述不完整
  位置：§二、P3 问题 5
  问题：P3 章节只点名 Weight Adam/AdamW 的 num_inputs < 5 无余量，但经代码验证，所有 9 个 CPU launcher 的 num_inputs < …
  均为恰好对齐：
   Launcher          Compiler 推入                num_inputs   守卫            状态
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Weight SGD        [lr, wd]                     2            2 < 2 = false   无余量
   Weight Momentum   [lr, wd, beta]               3            3 < 3 = false   无余量
   Weight Nesterov   [lr, wd, beta]               3            3 < 3 = false   无余量
   Weight Adam       [lr, wd, beta, beta2, eps]   5            5 < 5 = false   无余量
   Weight AdamW      [lr, wd, beta, beta2, eps]   5            5 < 5 = false   无余量
   Bias SGD          [lr]                         1            1 < 2 = true    已触发回归
   Bias Momentum     [lr, beta]                   2            2 < 2 = false   无余量
   Bias Nesterov     [lr, beta]                   2            2 < 2 = false   无余量
   Bias Adam         [lr, beta, beta2, eps]       4            4 < 4 = false   无余量
  建议：P3 章节开头补充一句：
  ▌ 经核查，全部 9 个 CPU launcher 的 num_inputs < N 守卫均为"恰好对齐"（无余量），Bias SGD 已触发回归（P0），其余 8
  ▌ 个为潜在回归隐患。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  结论
  除以上 2 处文档级不一致/遗漏 外，AXU_FINAL.md 的修复方案无其他问题：
  • ✅ P0 修复方案正确且最小化
  • ✅ P1 H2D 迁移逻辑正确（CUDA 路径 + CPU 路径）
  • ✅ P2 死代码清理清单完整
  • ✅ P3 系统性处理方案合理
  • ✅ 依赖顺序和执行流程正确

  建议接受，实施时顺带修正上述 2 处文档细节即可。