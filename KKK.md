# 【今日话题：卡死了？问题何在？】

各位。我们已经开发了非常厉害的算子。但是，曾经的模块Preprocessor的测试似乎受到了回归性影响，不能正常工作了。

最典型的就是tests/correction/test_pw_ultimate.cpp这个专门的测试样例。以前它在tests_legacy/prep/test_pw_ultimate.cpp里面。

旧版涉及include_legacy、src_legacy、tests_legacy三个目录。

而新版设计include、src、tests三个目录。

旧版的结果如根目录下的ksl_legacy.txt所示。

新版的结果如根目录下的ksl.txt所示。新版卡死了，没有后续输出，等多长时间也等不到第一个epoch的完成。

按理说Preprocessor与GPU上的各种算子是不同的体系，不应该受到我们后续开发的影响。但应该是某个小伙伴在开发时引入了回归性错误，导致Preprocessor的重要测试样例也跑不通了。

你的任务就是对比检查，看新版的preprocessor在跑这个测试样例时跟旧版到底有何不同，为什么会出现卡死，应该怎么修复。

我需要提及的两点就是：第一，TransferStation的缓冲区有可读可写标志位，未设置可写是不能写入的，因为GPU那边没有取走数据，但是这个可以通过设置FOR_TRANSFER_STATION_UNIT_TESTS_ONLY宏来解决，按理说现在已经在tests/correction/CMakeLists.txt设置了。第二，我们在开发过程中修改了TransferStation的容量计算，这个修改应该是必要的，因为需要与显存上的区域对齐。但是这个修改应该不至于直接导致无法运行。

请检查，阅读相关文件，分析原因，给出方案。