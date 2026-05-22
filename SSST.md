# 【小伙伴S】

  问题在于src指针计算！

  看失败的B区调试信息：
  staging_base=0x7a0409920c00 per_zone=1536 label_aligned=256
  range[0] start_region=51 (I_B_LABEL): src=0x7a0409921200
  range[1] start_region=52 (I_B_DATA): src=0x7a0409921300

  计算验证：
  - I_B_LABEL: staging_base + per_zone = 0x7a0409920c00 + 1536 = 0x7a0409920c00 + 0x600 = 0x7a0409921200 ✅
  - I_B_DATA: staging_base + per_zone + label_aligned = 0x7a0409920c00 + 1536 + 256 = 0x7a0409920c00 + 0x700 =
    0x7a0409921300 ✅

  指针计算看起来是正确的！

  但是注意StagingBufferPool的大小：
  - block_size=3072，所以总大小是3072字节
  - I_B_DATA的src_end = 0x7a0409921300 + 1536 = 0x7a0409921300 + 0x600 = 0x7a0409921900
  - StagingBufferPool结束地址 = 0x7a0409920c00 + 3072 = 0x7a0409920c00 + 0xC00 = 0x7a0409921800

  越界了！I_B_DATA的src_end (0x7a0409921900) > StagingBufferPool结束地址 (0x7a0409921800)！

  问题找到了！B区的I_B_DATA传输会越界，导致cudaMemcpyAsync报"invalid argument"错误！

  根本原因：MemoryPlan计算I_B_DATA的size时，可能计算错了或者包含了不该包含的部分！

  让我检查为什么A区通过但B区失败：
  - A区：dst_off=512, dst_size=1536，总计2048字节 ✅ 在3072范围内
  - B区：dst_off=2560, dst_size=1536，总计4096字节 ❌ 超出3072范围

  MemoryPlan的region bounds计算有问题！B区的偏移和大小计算错误！

  这意味着MemoryPlan在计算I_B_DATA的bounds时，可能包含了不正确的内存范围。问题不在h2d_op.cpp，而在MemoryPlan的region
  bounds计算！



# 【用户补充】

shape都错了，傻瓜。我们的张量是NHWC，你看看你们申请DTensor的时候把C通道放第几个了？全部transfer测试样例都检查一下！

