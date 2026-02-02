以下全都是有预热的结果，Linux云服务器平台测试结果（并行智算云5090x8，INTEL(R) XEON(R) GOLD 6530，112核112线程，960GB内存）

测试完成时间：2026年2月3日06:58:04

## MNIST RAW

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 0.244    | 0.180             |
| partial | val   | 0.114    | 0.064             |
| fully   | train | 0.207    | 0.212             |
| fully   | val   | 0.128    | 0.057             |

## MNIST DTS

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 0.140    | 0.313             |
| partial | val   | 0.113    | 0.065             |
| fully   | train | 0.171    | 0.257             |
| fully   | val   | 0.109    | 0.067             |

## CIFAR-10 RAW

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 0.686    | 0.209             |
| partial | val   | 0.171    | 0.167             |
| fully   | train | 0.724    | 0.198             |
| fully   | val   | 0.172    | 0.166             |

## CIFAR-10 DTS

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 0.236    | 0.606             |
| partial | val   | 0.132    | 0.217             |
| fully   | train | 0.252    | 0.568             |
| fully   | val   | 0.145    | 0.197             |

## CIFAR-100 RAW

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 0.500    | 0.286             |
| partial | val   | 0.209    | 0.137             |
| fully   | train | 0.471    | 0.304             |
| fully   | val   | 0.211    | 0.136             |

## CIFAR-100 DTS

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 0.226    | 0.633             |
| partial | val   | 0.144    | 0.199             |
| fully   | train | 0.200    | 0.715             |
| fully   | val   | 0.121    | 0.237             |

## ImageNet RAW

|         |       | Time (s) | Throughput (GB/s) |
| ------- | ----- | -------- | ----------------- |
| partial | train | 2.459    | 55.640            |
| partial | val   | 0.483    | 12.932            |
| fully   | train | 12.941   | 10.572            |
| fully   | val   | 0.581    | 10.750            |

## ImageNet DTS LV0

|                      |       | Time (s) | Throughput (GB/s) |
| -------------------- | ----- | -------- | ----------------- |
| partial              | train | 31.373   | 4.367             |
| partial              | val   | 0.166    | 37.745            |
| partial (second run) | train | 1.008    | 135.913           |
| partial (second run) | val   | 0.177    | 35.399            |
| fully                | train | 16.968   | 8.074             |
| fully                | val   | 0.865    | 7.243             |

## ImageNet DTS LV3

|                      |       | Time (s) | Throughput (GB/s) |
| -------------------- | ----- | -------- | ----------------- |
| partial              | train | 10.142   | 13.508            |
| partial              | val   | 0.140    | 44.754            |
| partial (second run) | train | 0.511    | 268.102           |
| partial (second run) | val   | 0.168    | 37.295            |
| fully                | train | 5.557    | 24.654            |
| fully                | val   | 0.354    | 17.700            |