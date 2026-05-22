# 【今日话题：NAN的修复】



众所周知，我们的框架的MemoryPlan必定会留一个NaN标志位。我们的算子也会操作这个NaN标志位。然后我们有个NaN的数学正确性测试在tests/correction目录下。

现在要探讨的是：这个NaN标志位应该改为FP32还是INT32？如果要改，就必须统一改，要改哪些地方？



# 【小伙伴K】

• 从框架设计角度，NaN 标志位本质上是整数语义（0 = 无 NaN，1 = 有 NaN），不是浮点数学运算的产物。当前状态和设计正确性的对
  比：
     维度     当前状态         设计正确性
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DType     FP32             应该是 INT32
    Region    S_SCALAR_FP32    应该是 S_SCALAR_INT32
   内存语义   4 字节浮点标量   4 字节整数标量
   取值含义   0.0f / 1.0f      0 / 1
  框架本身支持这个修正：
  • DType 有 INT32
  • Region 有 S_SCALAR_INT32（060）
  • Tensor::data<int32_t>() 可用（模板直接 static_cast<T*>，无类型检查障碍）
  • transfer_to_rank / fetch_from_rank 是纯内存拷贝，不依赖 dtype
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  我的观点：应该修改
  理由：
  1. 类型即文档：INT32 比 FP32 更准确地表达了"这是整数标志位"的意图
  2. 避免维护陷阱：刚踩的坑（int 写入 + float 读取 = 位模式错配）就是类型不匹配的直接后果。INT32 从头到尾统一用 int*/int
     t*，不存在 IEEE 754 位模式歧义
  3. Region 命名自洽：S_SCALAR_INT32  region 存在却未被使用，而 S_SCALAR_FP32 被用于非浮点用途，属于资源错配
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    如果修改，需要改动的位置
    tests/correction/test_check_nan.cpp：
    // 原
    DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::FP32, Region::S_SCALAR_FP32);
    // 改为
    DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::INT32, Region::S_SCALAR_INT32);
    ...
    // 读取
    int flag_val = h_flag.data<int32_t>()[0];
    src/backend/ops/range/check_op.cpp/.cu： 把刚才的 float* 改回 int*（语义更正确）：
    • check_op.cu：volatile int* + *has_nan_flag = 1
    • check_op.cpp CUDA launcher：int* has_nan_ptr + cudaMemsetAsync(..., sizeof(int), ...)
    • check_op.cpp CPU launcher：int* nan_ptr + *nan_ptr = has_nan
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    要不要我现在就把测试代码和算子实现统一改成 INT32 + S_SCALAR_INT32？