



# 【测试样例多平台运行结果】



## Linux（A100 × 8）

```shell
root@aabe98bd044a:~/epfs/R/renaissance/build/bin/tests/top# ./example_axpy
Device: GPU [1, 2, 4, 7]
Random Seed: 0x18AC1EF1B6062CEA

========================================
GPU Configuration: 8-GPU system detected
Using GPUs [1,2,4,7] for 4-card parallel test
Rank Mapping:
  rank 0 -> physical GPU 1
  rank 1 -> physical GPU 2
  rank 2 -> physical GPU 4
  rank 3 -> physical GPU 7
Verifying results from all 4 rank(s):
  Rank 0 (physical GPU 1): PASS - all elements = 5
  Rank 1 (physical GPU 2): PASS - all elements = 5
  Rank 2 (physical GPU 4): PASS - all elements = 5
  Rank 3 (physical GPU 7): PASS - all elements = 5
----------------------------------------
Computation: C = alpha * A + B = 2 * 1 + 3 = 5
Final Result: PASS
========================================

root@aabe98bd044a:~/epfs/R/renaissance/build/bin/tests/top# ./example_axpy_cpu
Device: CPU
Random Seed: 0x18AC1EF2500AABF0

========================================
CPU Configuration: Forced CPU mode
Using CPU for computation (ignoring GPU availability)
Verifying results from CPU:
  CPU: PASS - all elements = 5
----------------------------------------
Computation: C = alpha * A + B = 2 * 1 + 3 = 5
Final Result: PASS
========================================

root@aabe98bd044a:~/epfs/R/renaissance/build/bin/tests/top# ./example_dual_graph
Device: GPU [1, 2, 4, 7]
Random Seed: 0x18AC1EF31B2CCD7A

========================================
Dual Graph Test: Compute + Transfer Overlap
========================================
GPU Configuration: 4 rank(s)
Physical GPUs [1,2,4,7]
Rank Mapping:
  rank 0 -> physical GPU 1
  rank 1 -> physical GPU 2
  rank 2 -> physical GPU 4
  rank 3 -> physical GPU 7
----------------------------------------
Compute (AXPY + ReLU): PASS
  A = -4, W = 3, alpha = 2
  temp = 2 * (-4) + 3 = -5
  result = ReLU(-5) = 0
  result[0] = 0 (expected 0)

Transfer (Multi-to-Multi): PASS
  Each rank received distinct data correctly:
    rank 0: 10.0
    rank 1: 11.0
    rank 2: 12.0
    rank 3: 13.0
========================================

root@aabe98bd044a:~/epfs/R/renaissance/build/bin/tests/top# ./example_dual_graph_cpu
Device: CPU
Random Seed: 0x18AC1EF3C31BBC8E

========================================
CPU Configuration: Forced CPU mode
Using CPU for computation (ignoring GPU availability)

========================================
Dual Graph Test: Compute + Transfer (CPU)
========================================
Device: CPU (single rank)
----------------------------------------
Compute (AXPY + ReLU): PASS
  A = -4, W = 3, alpha = 2
  temp = 2 * (-4) + 3 = -5
  result = ReLU(-5) = 0
  result[0] = 0 (expected 0)

Transfer (H2D): PASS
  CPU received data correctly: 10.0
========================================

```



## Linux（RTX 5090 × 1）

```shell
root@renaissance-v4-0-2-120gb-100m:~/epfs/R/renaissance/build/bin/tests/top# ./example_axpy
Device: GPU [0]
Random Seed: 0x18AC1EEFE1D577FB

========================================
GPU Configuration: 1-GPU(s) detected
Using GPU 0 for single-card test
Verifying results from all 1 rank(s):
  Rank 0: PASS - all elements = 5
----------------------------------------
Computation: C = alpha * A + B = 2 * 1 + 3 = 5
Final Result: PASS
========================================

root@renaissance-v4-0-2-120gb-100m:~/epfs/R/renaissance/build/bin/tests/top# ./example_axpy_cpu
Device: CPU
Random Seed: 0x18AC1EEFECB31C69

========================================
CPU Configuration: Forced CPU mode
Using CPU for computation (ignoring GPU availability)
Verifying results from CPU:
  CPU: PASS - all elements = 5
----------------------------------------
Computation: C = alpha * A + B = 2 * 1 + 3 = 5
Final Result: PASS
========================================

root@renaissance-v4-0-2-120gb-100m:~/epfs/R/renaissance/build/bin/tests/top# ./example_dual_graph
Device: GPU [0]
Random Seed: 0x18AC1EEFFDB8F74C

========================================
Dual Graph Test: Compute + Transfer Overlap
========================================
GPU Configuration: 1 rank(s)
----------------------------------------
Compute (AXPY + ReLU): PASS
  A = -4, W = 3, alpha = 2
  temp = 2 * (-4) + 3 = -5
  result = ReLU(-5) = 0
  result[0] = 0 (expected 0)

Transfer (Multi-to-Multi): PASS
  Each rank received distinct data correctly:
    rank 0: 10.0
========================================

root@renaissance-v4-0-2-120gb-100m:~/epfs/R/renaissance/build/bin/tests/top# ./example_dual_graph_cpu
Device: CPU
Random Seed: 0x18AC1EF007F2ADD2

========================================
CPU Configuration: Forced CPU mode
Using CPU for computation (ignoring GPU availability)

========================================
Dual Graph Test: Compute + Transfer (CPU)
========================================
Device: CPU (single rank)
----------------------------------------
Compute (AXPY + ReLU): PASS
  A = -4, W = 3, alpha = 2
  temp = 2 * (-4) + 3 = -5
  result = ReLU(-5) = 0
  result[0] = 0 (expected 0)

Transfer (H2D): PASS
  CPU received data correctly: 10.0
========================================

```



## Windows（RTX 4060 × 1）

```shell
PS R:\renaissance\build\windows-msvc-release\bin\tests\top> .\example_axpy
Device: GPU [0]
Random Seed: 0x000E7E51196D00F0

========================================
GPU Configuration: 1-GPU(s) detected
Using GPU 0 for single-card test
Verifying results from all 1 rank(s):
  Rank 0: PASS - all elements = 5
----------------------------------------
Computation: C = alpha * A + B = 2 * 1 + 3 = 5
Final Result: PASS
========================================

PS R:\renaissance\build\windows-msvc-release\bin\tests\top> .\example_axpy_cpu
Device: CPU
Random Seed: 0x000E7E51DF3A439C

========================================
CPU Configuration: Forced CPU mode
Using CPU for computation (ignoring GPU availability)
Verifying results from CPU:
  CPU: PASS - all elements = 5
----------------------------------------
Computation: C = alpha * A + B = 2 * 1 + 3 = 5
Final Result: PASS
========================================

PS R:\renaissance\build\windows-msvc-release\bin\tests\top> .\example_dual_graph
Device: GPU [0]
Random Seed: 0x000E7E531D9AAE14

========================================
Dual Graph Test: Compute + Transfer Overlap
========================================
GPU Configuration: 1 rank(s)
----------------------------------------
Compute (AXPY + ReLU): PASS
  A = -4, W = 3, alpha = 2
  temp = 2 * (-4) + 3 = -5
  result = ReLU(-5) = 0
  result[0] = 0 (expected 0)

Transfer (Multi-to-Multi): PASS
  Each rank received distinct data correctly:
    rank 0: 10.0
========================================

PS R:\renaissance\build\windows-msvc-release\bin\tests\top> .\example_dual_graph_cpu
Device: CPU
Random Seed: 0x000E7E542748ED58

========================================
CPU Configuration: Forced CPU mode
Using CPU for computation (ignoring GPU availability)

========================================
Dual Graph Test: Compute + Transfer (CPU)
========================================
Device: CPU (single rank)
----------------------------------------
Compute (AXPY + ReLU): PASS
  A = -4, W = 3, alpha = 2
  temp = 2 * (-4) + 3 = -5
  result = ReLU(-5) = 0
  result[0] = 0 (expected 0)

Transfer (H2D): PASS
  CPU received data correctly: 10.0
========================================

```



