**测试平台：**智星云GPU服务器（CPU：Intel Xeon Platinum 8480C × 2，112核224线程；内存：512GB；GPU：A100 × 8；显存：40GB × 8；操作系统：Ubuntu 24.04）

**测试日期：**2026年2月9日

**测试线程配置：**16线程加载，96线程预处理



## 无缓存预热的实验结果

**（单位：s）**

|  Dataset  | Format  |  Mode   | TRAIN  |  VAL  | TRAIN+VAL |
| :-------: | :-----: | :-----: | :----: | :---: | :-------: |
|   MNIST   |   RAW   |  FULLY  | 0.203  | 0.122 |   0.325   |
|   MNIST   |   DTS   |  FULLY  | 0.168  | 0.113 |   0.281   |
| CIFAR-10  |   RAW   |  FULLY  | 0.701  | 0.204 |   0.905   |
| CIFAR-10  |   DTS   |  FULLY  | 0.273  | 0.135 |   0.408   |
| CIFAR-100 |   RAW   |  FULLY  | 0.532  | 0.200 |   0.732   |
| CIFAR-100 |   DTS   |  FULLY  | 0.243  | 0.131 |   0.374   |
| ImageNet  |   RAW   |  FULLY  | 92.648 | 4.252 |  96.900   |
| ImageNet  | DTS LV0 |  FULLY  | 76.632 | 3.607 |  80.239   |
| ImageNet  | DTS LV1 |  FULLY  | 35.220 | 1.602 |  36.822   |
| ImageNet  | DTS LV2 |  FULLY  | 35.033 | 1.616 |  36.649   |
| ImageNet  | DTS LV3 |  FULLY  | 24.620 | 1.147 |  25.767   |
| ImageNet  |   RAW   | PARTIAL | 91.593 | 4.238 |  95.831   |
| ImageNet  | DTS LV0 | PARTIAL | 75.726 | 3.571 |  79.297   |
| ImageNet  | DTS LV1 | PARTIAL | 35.890 | 1.687 |  37.577   |
| ImageNet  | DTS LV2 | PARTIAL | 36.249 | 1.635 |  37.884   |
| ImageNet  | DTS LV3 | PARTIAL | 25.408 | 1.163 |  26.571   |



## 有缓存预热的实验结果

**（单位：s）**

|  Dataset  | Format  |  Mode   | TRAIN  |  VAL  | TRAIN+VAL |
| :-------: | :-----: | :-----: | :----: | :---: | :-------: |
|   MNIST   |   RAW   |  FULLY  | 0.200  | 0.124 |   0.324   |
|   MNIST   |   DTS   |  FULLY  | 0.151  | 0.110 |   0.261   |
| CIFAR-10  |   RAW   |  FULLY  | 0.606  | 0.178 |   0.784   |
| CIFAR-10  |   DTS   |  FULLY  | 0.236  | 0.124 |   0.360   |
| CIFAR-100 |   RAW   |  FULLY  | 0.486  | 0.187 |   0.673   |
| CIFAR-100 |   DTS   |  FULLY  | 0.237  | 0.123 |   0.360   |
| ImageNet  |   RAW   |  FULLY  | 16.369 | 0.708 |  17.077   |
| ImageNet  | DTS LV0 |  FULLY  | 13.757 | 0.679 |  14.436   |
| ImageNet  | DTS LV1 |  FULLY  | 7.422  | 0.371 |   7.793   |
| ImageNet  | DTS LV2 |  FULLY  | 6.516  | 0.355 |   6.871   |
| ImageNet  | DTS LV3 |  FULLY  | 4.640  | 0.277 |   4.917   |
| ImageNet  |   RAW   | PARTIAL | 6.313  | 0.723 |   7.036   |
| ImageNet  | DTS LV0 | PARTIAL | 4.399  | 0.307 |   4.706   |
| ImageNet  | DTS LV1 | PARTIAL | 2.099  | 0.182 |   2.281   |
| ImageNet  | DTS LV2 | PARTIAL | 2.195  | 0.195 |   2.390   |
| ImageNet  | DTS LV3 | PARTIAL | 1.645  | 0.161 |   1.806   |



从上表可见，DTS相对于RAW有平均 **1.5×** 的加速比。