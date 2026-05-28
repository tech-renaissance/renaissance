# 【小伙伴K】

  总体评价
  Fix-1（P0）和 Fix-2（P1）完全正确，能解决问题，无问题。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  发现的问题
  Fix-3（P2）存在一处编译可见性问题：
  FVB_FINAL.md 建议的调试修复代码使用了 half_to_float：
  float v = half_to_float(fout16[i * 4]);
  但 half_to_float 定义在 src/tensor/tensor.cpp:1540，且为 static 函数：
  static float half_to_float(uint16_t fp16) { ... }
  static 函数仅在定义文件内可见，preprocess_worker.cpp 位于 src/data/ 目录，无法直接调用 tensor.cpp 的 static 函数，直接
  复制该代码会导致编译失败。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  修改意见
  建议将 Fix-3 改为以下两种方案之一：
  方案 A：在 DIAG 块内内联简单转换（推荐，最小侵入）
  在 preprocess_worker.cpp 的 DIAG-WORKER 块内，直接写一个局部 lambda 做 FP16→FP32，不依赖外部函数：
  if (using_amp_) {
      const std::uint16_t* fout16 = reinterpret_cast<const std::uint16_t*>(final_output_ptr);
      auto half_to_float_local = [](uint16_t h) -> float {
          uint32_t sign = (h >> 15) & 0x1;
          uint32_t exp  = (h >> 10) & 0x1F;
          uint32_t mant = h & 0x3FF;
          if (exp == 0) return sign ? -0.0f : 0.0f;
          if (exp == 31) return (mant == 0)
              ? (sign ? -std::numeric_limits<float>::infinity()
                      : std::numeric_limits<float>::infinity())
              : std::numeric_limits<float>::quiet_NaN();
          return (sign ? -1.0f : 1.0f) * std::pow(2.0f, (float)exp - 15.0f)
                 * (1.0f + (float)mant / 1024.0f);
      };
      for (int i = 0; i < 784; ++i) {
          float v = half_to_float_local(fout16[i * 4]);
          ...
      }
  }
  方案 B：提取公共辅助函数
  将 tensor.cpp 中的 half_to_float 提取到 include/renaissance/core/types.h 或新建 include/renaissance/core/fp16_utils.h
  中作为 inline 函数，供全项目使用。
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  补充说明
  • Fix-1 和 Fix-2 的 SIMD 指令组合经过验证，布局完全正确，无需修改。
  • FVB_FINAL.md Fix-2 说明中的笔误 "共删 3 行、减 2 行" 应为 "共删 3 行"，不影响执行。
  • Fix-3 本身是 P2（建议级），不改不影响生产数据，但若执行需先解决可见性问题。