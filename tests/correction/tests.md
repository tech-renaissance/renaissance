基础测试

```shell
/root/epfs/R/renaissance/build/bin/tests/correction/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --batch-size 512 --resolution 224 --loaders 16 --preproc 128 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true --reproducible --cpu-bind --amp



/root/epfs/R/renaissance/build/bin/tests/correction/perf_h2d_copy_a --cpu

/root/epfs/R/renaissance/build/bin/tests/correction/perf_h2d_copy_a --gpu

/root/epfs/R/renaissance/build/bin/tests/correction/perf_h2d_copy_a --amp



/root/epfs/R/renaissance/build/bin/tests/correction/perf_h2d_copy_b --cpu

/root/epfs/R/renaissance/build/bin/tests/correction/perf_h2d_copy_b --gpu

/root/epfs/R/renaissance/build/bin/tests/correction/perf_h2d_copy_b --amp



/root/epfs/R/renaissance/build/bin/tests/correction/test_h2d_copy_a --cpu

/root/epfs/R/renaissance/build/bin/tests/correction/test_h2d_copy_a --gpu

/root/epfs/R/renaissance/build/bin/tests/correction/test_h2d_copy_a --amp



/root/epfs/R/renaissance/build/bin/tests/correction/test_h2d_copy_b --cpu

/root/epfs/R/renaissance/build/bin/tests/correction/test_h2d_copy_b --gpu

/root/epfs/R/renaissance/build/bin/tests/correction/test_h2d_copy_b --amp



/root/epfs/R/renaissance/build/bin/tests/perf/test_gap_fc_perf --amp

```
