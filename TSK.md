# 【今日话题：仅测量H2D传输的方案】

给DeepLearningTask添加两个方法：
compile_h2d_only()——所有图集中，只编译和捕获H2D传输相关的图（一个A区一个B区），包括last batch。不过，不需要捕获验证的
run_h2d_only()——只交替运行H2D传输相关的图（一个A区一个B区），需要启动Preprocessor的train，只执行一个epoch

然后写一个测试样例，基于test_pw_ultimate.cpp，但是要加上DeepLearningTask的compile_h2d_only()和run_h2d_only()

BluePrint的定义写简单一点，就是直接一个FC，输出神经元个数等于类别数即可（毕竟我们不追求准确率）。

我们的目的是什么呢？测试Preprocessor与异步传输对接后的每epoch耗时。
再次强调，只跑H2D传输图即可。
大致的流程就是，preprocessor往TransferStation的A区填充数据，然后设可读标志，RANK线程看到可读标志后就启动H2D传输图当中的A图，传输完成就设可写，然后B区，如此往复，直到train数据集预处理并传输完成。
注意，只处理一个epoch的train数据集即可。需要妥善处理last batch
需要对run_h2d_only()计时。
你的run_h2d_only()方法必须没有多余的动作，否则会拖慢运行时间。另外，A区的传输范围是整个I_A_LABEL和I_A_DATA，B区的传输范围是整个I_B_LABEL和I_B_DATA

现有的测试样例基本都不满足要求，但perf_h2d_copy_a.cpp和perf_h2d_copy_b.cpp可以提供一定参考。（它们用的是SimpleTask，但现在我们要用DeepLearningTask）

请认真研究当前实现，然后给出方案。





