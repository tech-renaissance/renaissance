---
##  一、PO情况

| 序号 | PreprocessOperation       |
| ---- | ------------------------- |
| 1    | Resize                    |
| 2    | CenterCrop                |
| 3    | RandomResizedCrop         |
| 4    | RandomHorizontalFlip      |
| 5    | ColorJitter               |
| 6    | RandomRotation            |
| 7    | RandomGrayscale           |
| 8    | GaussianBlur              |
| 9    | RandomCrop                |
| 10   | RandomAutocontrast        |
| 11   | RandomErasing (待GPU实现) |
| 12   | FastRandomResizedCrop     |
| 13   | GaussianNoise             |
| 14   | Pad                       |
| 15   | RandomBrightness          |



## 二、PO组合性能测试

由于一些操作要求固定尺寸输入，所以不能单独测试，只能与其他PO（如CenterCrop）组合测试

### 测试命令（示例）：

```shell
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 FastRandomResizedCrop --po-val2 RandomHorizontalFlip --seed 42 --sdmp 1 --cpvs false

```



### 测试结果：

| 序号 | 操作组合                                   | 总时间（Train+Val） |
| ---- | ------------------------------------------ | ------------------- |
| 1    | CenterCrop+RandomHorizontalFlip            | 23.686              |
| 2    | CenterCrop+DoNothing                       | 22.819              |
| 3    | RandomCrop+RandomHorizontalFlip            | 44.890              |
| 4    | RandomCrop+DoNothing                       | 44.401              |
| 5    | RandomResizedCrop+RandomHorizontalFlip     | 28.265              |
| 6    | RandomResizedCrop+DoNothing                | 27.740              |
| 7    | FastRandomResizedCrop+RandomHorizontalFlip | 28.180              |
| 8    | FastRandomResizedCrop+DoNothing            | 27.556              |
| 9    | Resize+RandomHorizontalFlip                | 46.231              |
| 10   | Resize+CenterCrop                          | 45.434              |
| 11   | Resize+DoNothing                           | 45.365              |
| 12   | CenterCrop+RandomGrayscale                 | 23.411              |
| 13   | CenterCrop+GaussianBlur                    | 24.501              |
| 14   | CenterCrop+ColorJitter                     | 46.209              |
| 15   | CenterCrop+RandomAutocontrast              | 28.322              |
| 16   | CenterCrop+RandomRotation                  | 39.374              |
| 17   | CenterCrop+GaussianNoise                   | 37.280              |
| 18   | CenterCrop+Pad                             | 24.597              |
| 19   | CenterCrop+RandomBrightness                | 25.130              |



### Resize/Crop类速度排行：

（注：这里的用时其实是加载+解码+PO的总用时）

| 排名 | PO                    | 用时（仅供参考） |
| ---- | --------------------- | ---------------- |
| 1    | CenterCrop            | 22.819           |
| 2    | FastRandomResizedCrop | 27.556           |
| 3    | RandomResizedCrop     | 27.740           |
| 4    | RandomCrop            | 44.401           |
| 5    | Resize                | 45.365           |



### 非Resize/Crop类速度排行：

| 排名 | PO                   | 用时（仅供参考） |
| ---- | -------------------- | ---------------- |
| 1    | DoNothing            | 0.000            |
| 2    | RandomGrayscale      | 0.592            |
| 3    | RandomHorizontalFlip | 0.867            |
| 4    | GaussianBlur         | 1.682            |
| 5    | Pad                  | 1.778            |
| 6    | RandomBrightness     | 2.311            |
| 7    | RandomAutocontrast   | 5.503            |
| 8    | GaussianNoise        | 14.461           |
| 9    | RandomRotation       | 16.555           |
| 10   | ColorJitter          | 23.390           |



## 三、FRRC Vs RRC

### 测试命令：

```shell
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true

/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true

/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true

/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true

```



### 测试结果：

|                     | Train  |  Val  | Train+Val |
| :-----------------: | :----: | :---: | :-------: |
| **RRC** (SDMP = 1)  | 26.766 | 0.304 |  27.070   |
| **FRRC** (SDMP = 1) | 26.502 | 0.315 |  26.817   |
| **RRC** (SDMP = 2)  | 22.504 | 0.296 |  22.801   |
| **FRRC** (SDMP = 2) | 16.966 | 0.296 |  17.262   |

