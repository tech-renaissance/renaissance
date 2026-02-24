测试日期：2026年2月22日

测试版本：**V3.20.2**

测试命令：

```shell
/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs false

/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true

/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs false

/root/epfs/R/renaissance/build/bin/tests/integration/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --cpu-bind --batch-size 512 --resolution 224 --loaders 16 --preproc 112 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 10 --po-train1 RandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 2 --cpvs true
```

测试结果：

|             | Train  |  Val  | Train+Val |
| :---------: | :----: | :---: | :-------: |
|  Baseline   | 26.827 | 2.055 |  28.882   |
|    CPVS     | 26.732 | 0.304 |  27.036   |
|    SDMP     | 22.586 | 2.045 |  24.632   |
| SDMP + CPVS | 22.545 | 0.301 |  22.846   |