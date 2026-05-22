# 【今日话题：如何优化框架的CUDA Graph执行逻辑】

# 【用户提供的指导思路】

**（我的观点不一定对，可能有很多笔误，但我绝对想到了很多小伙伴没想到的点，希望大家认真阅读，认真思考！）**

基本原则：
理论上每个epoch的训练每个线程只需要进行一次setdevice，不需要多次
DeepLearningTask除非是到了epoch结束，否则不要调用DeviceSync，而是对指定的流进行同步。同步计算流时，一定要三个流都同步。
尽可能避免在每个batch的循环中检查这个检查那个
一定要在正式训练开始前完成预热、图捕获和实例化！是的，每张卡上的实例化也要完成！千万千万不能等到run再来实例化，那成本可就高了
流的创建也是要在预热前完成，创建了就不要动了，保存好指针。注意，我们只有更新流（stream_update），没有通信流（stream_comm）
每个rank一个线程，线程id就是rank id，进入每个epoch的训练会展开多线程，结束这个epoch的训练就join
进入每个epoch的验证会展开多线程，结束这个epoch的验证就join
还是要再强调一遍，实例化要在用户执行DeepLearningTask::run()之前完成，否则会很花时间
我们的思路是把必要的实例化图准备好，然后通过某个索引方法来根据ShapeId（代表是哪种输入形状）、GraphId（代表是哪种图）来找到它
但是这个找的过程，我们都是在epoch展开多线程后一次性找好，记住它的指针。后面直接解引用就可以跑这个图了
实例化的时候应该也是要展开多线程，毕竟每个线程对应一张卡，每张卡对应自己的图集
我不知道是不是可以用std::vector<std::vector<cudaGraphExec_t> >来存放？第一次索引是RANK，第二次索引是具体的图编号exec_id
总之每个RANK会有它的一套图，事先捕获好、实例化好

求解索引的方法——get_cuda_graph()，参数是：ShapeId（代表是哪种输入形状）、GraphId（代表是哪种图）





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


cuda_graph_ptr_transfer_a->run(); // 我不知道CUDA Graph是不是这样用的，大概这个意思
sync_transfer_stream();

// batch迭代
for (int batch_id = 0; batch_id != num_total_batches - 1; ++batch_id) {  // 减一是为了排除最后不完整batch

// 进入batch，先设置学习率，方法是对CUDA GRAPH setparam，最好不要用H2D

cudaGraphExec_t* cuda_graph_ptr_transfer_overlap = (batch_id % 2 == 0)? cuda_graph_ptr_transfer_b: cuda_graph_ptr_transfer_a;  // 选定需要被并行掩盖的那个传输图
cudaGraphExec_t* cuda_graph_ptr_first_overlap = (batch_id % 2 == 0)? cuda_graph_ptr_first_layer_from_a: cuda_graph_ptr_first_layer_from_b;  // 选定需要被并行掩盖的那个传输图


cudaGraphLaunch(*cuda_graph_ptr_transfer_overlap, *stream_comp_1_ptr);  //

cudaGraphLaunch(*cuda_graph_ptr_first_overlap, *stream_comp_1_ptr);  // 双图并行。多计算流捕获的图，用计算流1来启动
cudaGraphLaunch(*cuda_graph_ptr_zero_grad, *stream_update_ptr);  // 双图并行。与梯度有关的基本都是更新流在管

cudaStreamSynchronize(*stream_comp_1_ptr);  // 注意！三计算流同步！
cudaStreamSynchronize(*stream_comp_2_ptr);
cudaStreamSynchronize(*stream_comp_3_ptr);
cudaStreamSynchronize(*stream_update_ptr);

cudaGraphLaunch(*cuda_graph_ptr_deep_fwd_bwd, *stream_comp_1_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_transfer_overlap, *stream_trans_ptr);  // 双图并行

cudaStreamSynchronize(*stream_comp_1_ptr);  // 注意！三计算流同步！
cudaStreamSynchronize(*stream_comp_2_ptr);
cudaStreamSynchronize(*stream_comp_3_ptr);
cudaStreamSynchronize(*stream_trans_ptr);

cudaGraphLaunch(*cuda_graph_ptr_deep_grad_convert_fp16_to_fp32, *stream_update_ptr);
cudaStreamSynchronize(*stream_update_ptr);

cudaGraphLaunch(*cuda_graph_ptr_first_bwd, *stream_comp_1_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_deep_comm, *stream_update_ptr);  // 双图并行

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







而对于我们的SimpleTask，我的思路也是一样的：你要在run之前就完成好实例化，然后run只是根据图名来找到图然后跑而已
我建议提供一个run_iter("", "", int iterations)方法，获取了要跑的2张实例化图之后，直接进行循环，连解哈希都不放进循环里

我的SimpleTask的run_iter就只有：


void run_iter(const std::string& graph_name_1, const std::string& graph_name_2, int iterations) {

// 展开多线程，每个rank一个线程

cudaGraphExec_t* graph_ptr_1 = ...;  // 按字符串或ID找到图
cudaGraphExec_t* graph_ptr_2 = ...;  // 按字符串或ID找到图
cudaStream_t* stream_1 = ...;  // 按字符串或ID找到对应的流
cudaStream_t* stream_2 = ...;  // 按字符串或ID找到对应的流

for (int i = 0; i != iterations; ++i) {
cudaGraphLaunch(*graph_ptr_1, *stream_1);
cudaGraphLaunch(*graph_ptr_2, *stream_2);

cudaStreamSynchronize(*stream_1); // 并行运行完再一起同步
cudaStreamSynchronize(*stream_2);

}

// 多线程join

}

这里还有一个疑点就是，
cudaStream_t有必要传指针吗？如果它本身也就是一个整数的大小，那就没必要传指针解引用了。你们核实一下

说了那么多，我想说明什么呢？——你要尽可能地缩减DeepLearningTask训练验证循环或SimpleTask里的同样的图反复执行的循环里的多余操作
你不要在循环里找这个找那个，配置这个配置那个
最最最最重要的，是你不要在run里面实例化。我们的run前面是compile，compile阶段把能做的事都做好，把图实例化后放好，通过一个简单的索引就能找到，而且找的过程也尽可能不要放在循环里，要在循环外事先找好，然后再跑
这样一来，我们就把跑图的性能尽可能地最优化了，确保run里面的时间基本上都是DEVICE端在忙，而我们HOST不要有太多的查找、配置的复杂逻辑

我想到的一个我们的性能退化的原因就是，我们在SimpleTask的“重复运行同一个算子测性能”的测试里面，把多线程的展开和join放进了循环里。只要提供了一个run_iter方法来专门测单算子性能，不就可以避免这个问题了？多线程的展开就在测试次数迭代之外了