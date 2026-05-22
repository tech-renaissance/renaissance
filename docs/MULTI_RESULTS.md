# 三流架构最终测试数据（A100）



## 单流

```shell

GAP+FC(/root/epfs/R/renaissance/build/bin/tests/perf/test_gap_fc_perf):
===== 性能汇总 =====
  FWD (GAP+FC):      132.92 μs/iter
  BWD (FC_BWD+GAP_BWD): 481.65 μs/iter
  总计:             614.57 μs/iter

```



## 三流

```shell
GAP(/root/epfs/R/renaissance/build/bin/tests/perf/test_gap_perf):
===== 性能汇总 =====
  FWD: 87.97 μs/iter
  BWD: 407.55 μs/iter
  总计: 495.52 μs/iter


FC(/root/epfs/R/renaissance/build/bin/tests/op/test_fc_amp):
===== 性能汇总 =====
  FWD: 50.64 μs/iter
  BWD: 56.62 μs/iter
  总计: 107.26 μs/iter


GAP+FC(/root/epfs/R/renaissance/build/bin/tests/perf/test_gap_fc_perf):
===== 性能汇总 =====
  FWD (GAP+FC):      131.18 μs/iter
  BWD (FC_BWD+GAP_BWD): 456.05 μs/iter
  总计:             587.23 μs/iter


```

