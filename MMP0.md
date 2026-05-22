# 【今日话题：把MLP从dry run变为真实训练】



# 【背景】

各位小伙伴。

我们的深度学习框架终于到了正式训练MLP的这一步了。

至少我们已经验证过，核心的算子都能work了。比如tests/correction/test_mlp_final.cpp验证了所有层的正向反向算子，优化器是在test_sgd_weight.cpp、test_sgd_bias.cpp，传输是在test_h2d_copy_a.cpp（从A区传输）和test_h2d_copy_b.cpp（从B区传输）和test_h2d_copy_dtensor.cpp（传输单个FP32值，一般用于学习率更新）。

此外我们还有已经通过了验证的学习率调度器Scheduler和初始化器Initializer。

现在我们就是要把tests/ref/mnist_mlp_3.cpp的dry run改为真run。

dry run就是compile_for_dry_run()+dry_run()的组合，而真run是compile()+run()。

现在DeepLearningTask的这两个方法显然还没有完整实现。必须说明的是：DeepLearningTask的这两个方法要跟SimpleTask完全区分开，不应该共享。



下面我说一下我思路中的流程。

compile的时候，必须在compile_for_dry_run的基础上，完成warmup，完成所有必要的CUDA Graph的捕获和实例化，它需要执行完类似于SimpleTask的compile的所有事情。注意需要支持多RANK。我们的test_h2d_copy_a.cpp就能很好地支持多RANK。compile阶段还要完成所有数据的初始化。

而run就是按照一定的顺序运行各个CUDA Graph，其中有一些是并行的，但并行执行的CUDA Graph最多不超过2个。



数据的初始化，你需要先对整个显存池（ArenaKeeper）的所有RANK进行memset清零，然后调用初始化器（内部原理是Tensor初始化后同步传输并广播的形式传给所有RANK的DTensor），对所有DTensor进行初始化。

注意，一定要提前分配好学习率张量（这个应该已经实现了）。



完成compile之后，进入run，它需要展开N+1个线程，N是RANK数。0~N-1号线程就负责0~N-1号RANK。而第N号线程是负责去启动Preprocessor的train()和val()的。

这里说一下，Preprocessor可能会展开上百个线程来做预处理，所以负责RANK的那N个线程必须节省资源。

我的思路是让负责RANK管理的N个线程用std::condition_variable::wait()来等待Preprocessor把TransferStation的A区填满，看到可读标志，然后就触发从A区传输的图。触发之后，可以考虑用cudaStreamSynchronize来等待传输流传输完成，这个操作似乎不会占用CPU线程的资源。从TransferStation的A区传输完成后，要把TransferStation的A区标为可写。可读可写标志位是原子的。



按我的初步思路，是这样的（以下是伪代码）：



```c++
// epoch开始
// 展开多线程，进入epoch
// 每个rank一个线程，线程id就是rank id

// 线程初始化
cudaSetDevice(get_gpu_id_by_rank(rank_id));  // set一次，跑全部图，无需多次setDevice！

// 局部变量存好所有需要的CUDA Graph的指针
cudaGraphExec_t*  cuda_graph_ptr_transfer_a = get_cuda_graph(shape_id_1, TRANSFER_A);
cudaGraphExec_t*  cuda_graph_ptr_transfer_b = get_cuda_graph(shape_id_2, TRANSFER_B);
cudaGraphExec_t*  ...... // 所有图都各用一个局部指针变量存好，对于每个RANK来说，总共也就十几二十张图而已
// 这里我们几乎是一次性地把当前shape的所有图准备好了，循环内连索引都不需要，基本上就是解引用而已


cudaGraphLaunch(*cuda_graph_ptr_transfer_overlap, stream_comp_1_ptr); // 我不知道CUDA Graph是不是这样用的，大概这个意思
cudaStreamSynchronize(stream_trans_ptr);  // 这个一定要同步，确保传输完成

// batch迭代
for (int batch_id = 0; batch_id != num_total_batches - 1; ++batch_id) {  // 减一是为了排除最后不完整batch

cudaGraphExec_t* cuda_graph_ptr_transfer_overlap = (batch_id % 2 == 0)? cuda_graph_ptr_transfer_b: cuda_graph_ptr_transfer_a;  // 选定需要被并行掩盖的那个传输图
cudaGraphExec_t* cuda_graph_ptr_first_overlap = (batch_id % 2 == 0)? cuda_graph_ptr_first_layer_from_a: cuda_graph_ptr_first_layer_from_b;  // 选定需要被并行掩盖的那个传输图

cudaGraphLaunch(*cuda_graph_ptr_first_overlap, stream_comp_1_ptr);  // 双图并行。多计算流捕获的图，似乎是用主流来启动，这个你们仔细看SimpleTask的做法
cudaGraphLaunch(*cuda_graph_ptr_zero_grad, stream_update_ptr);  // 双图并行。与梯度有关的基本都是更新流在管

cudaStreamSynchronize(stream_comp_1_ptr);  // 注意！三计算流同步！
cudaStreamSynchronize(stream_comp_2_ptr);
cudaStreamSynchronize(stream_comp_3_ptr);
cudaStreamSynchronize(stream_update_ptr);

cudaGraphLaunch(*cuda_graph_ptr_deep_fwd_bwd, stream_comp_1_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_transfer_overlap, stream_trans_ptr);  // 双图并行

cudaStreamSynchronize(stream_comp_1_ptr);  // 注意！三计算流同步！
cudaStreamSynchronize(stream_comp_2_ptr);
cudaStreamSynchronize(stream_comp_3_ptr);
cudaStreamSynchronize(stream_trans_ptr);

cudaGraphLaunch(*cuda_graph_ptr_deep_grad_convert_fp16_to_fp32, stream_update_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_transfer_learning_rate, stream_update_ptr);  // 双图并行
cudaStreamSynchronize(*stream_update_ptr);
cudaStreamSynchronize(*stream_trans_ptr);

cudaGraphLaunch(*cuda_graph_ptr_first_bwd, stream_comp_1_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_deep_comm, stream_update_ptr);  // 双图并行

sync_three_compute_streams();
sync_comm_streams();

... // 其他图
// 检测NaN后，通过CUDA Graph的条件节点来判断要不要执行权重更新相关的图

}

// 这里是last batch的逻辑，last batch不需要传输了，因为上个batch已经传输

auto cuda_graph_ptr_last_batch_fisrt_layer = (num_total_batches % 2 == 0)? cuda_graph_ptr_first_layer_from_b: cuda_graph_ptr_first_layer_from_a;  // 最后一个batch从何处开始

cuda_graph_ptr_last_batch_fisrt_layer->run();  // 双图并行
cuda_graph_ptr_zero_grad->run();  // 双图并行。与梯度相关的基本都是更新流在管

sync_three_compute_streams();
sync_update_streams();

... // 后面类似

// epoch的train阶段结束

// 线程join

// 取回epoch train结果
```



具体的写法，要参照SimpleTask的，我上面只是伪代码。但SimpleTask至少是跑通了的。

问题在于，我们的DeepLearningTask跑的图集与SimpleTask不完全相同，图集是可以根据输入分辨率的不同来选择的。所以需要看清楚。

我想强调的是：第一个batch的传输，必定是从TransferStation的A区传递到I_A_LABEL和I_A_DATA。后面是AB区之间乒乓切换。

控制流你们需要再检查优化一下，因为如果不开启AMP的话，是完全不需要用到FP32和FP16之间的切换的。如果图集里的是空图，显然就应该跳过执行。

最关键的重叠就是传输与深层FWD/BWD的重叠，其次是首层反向传播和深层通信的重叠。

我强烈建议**分步实现**，第一步毫无疑问就是正确地实现A区的异步传输，确保跟Preprocessor成功对接并传输有效数据到正确的地方（比如，把TransferStation的前几个数打印出来，然后Transfer之后从I_A_DATA或I_A_LABEL把DTensor给fetch回来也打印前几个数进行对比）；第二步则是把AB区传输运转起来，实现单epoch内的所有train和val的传输；后续再验证其他图。

再次强调，选择图集是需要看分辨率的，好像是通过ShapeId实现。



请大家认真调研，给出方案。



























