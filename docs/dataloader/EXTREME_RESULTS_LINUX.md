测试平台：Linux，112核 960GB内存

测试配置：16线程加载

### DTL Loader用时（单位：s）

|               | Cold Cache | Warm Cache |
| ------------- | ---------- | ---------- |
| PARTIAL VAL   | 3.503      | 0.343      |
| FULLY VAL     | 21.332     | 1.238      |
| PARTIAL TRAIN | 31.022     | 1.166      |
| FULLY TRAIN   | 467.759    | 18.207     |



### DTL Loader吞吐量（单位：GB/s）

|               | Cold Cache | Warm Cache |
| ------------- | ---------- | ---------- |
| PARTIAL VAL   | 1.789      | 18.294     |
| FULLY VAL     | 0.294      | 5.061      |
| PARTIAL TRAIN | 4.416      | 117.496    |
| FULLY TRAIN   | 0.293      | 7.525      |



### RAW Loader用时（单位：s）

|               | Cold Cache | Warm Cache |
| ------------- | ---------- | ---------- |
| PARTIAL VAL   | 25.113     | 1.001      |
| FULLY VAL     | 24.767     | 1.058      |
| PARTIAL TRAIN | 214.470    | 3.572      |
| FULLY TRAIN   | 543.480    | 11.844     |



### RAW Loader吞吐量（单位：GB/s）

|               | Cold Cache | Warm Cache |
| ------------- | ---------- | ---------- |
| PARTIAL VAL   | 0.249      | 6.243      |
| FULLY VAL     | 0.252      | 5.904      |
| PARTIAL TRAIN | 0.638      | 38.303     |
| FULLY TRAIN   | 0.252      | 11.552     |

