你现在看到的UnifiedDataLoader是我们旧版的方案。设计这个类是为了减少配置不同DataLoader的复杂度、统一API。它的基本思路是，设计一个同为DataLoader子类的UnifiedDataLoader，用来跟Preprocessor对接，内部保存其他6个DataLoader之一的指针，然后通过这个类来配置对应的具体的DataLoader，并且在其后转发Preprocessor的一切操作给具体的DataLoader（比如ImageNetLoaderRaw）。但现在这个实现的问题是，对于ImageNet RAW，它的速度**始终比ImageNetLoaderRaw慢40%**。可能热路径上要执行几百万次操作，所以解引用、压栈、虚函数表等都会成为时间开销。我们尝试了很多方式来优化转发，但效果都不好，问题可能还出现在缓存友好性上。

现在我们改变了思路，就是直接取消UnifiedDataLoader这个类，确保Preprocessor直接对接具体的DataLoader，但是，要在Preprocessor类内部，借鉴UnifiedDataLoader的configure方法，来实现对具体DataLoader的**一次性的统一配置**。因为这个方法可以配置所有6种DataLoader。我们的做法是，用户不再亲自操作具体DataLoader的配置，而是通过Preprocessor的config_dataloader方法来完成。Preprocessor内置对具体DataLoader的引用（或指针），无需用户亲自在测试样例中绑定。用户只需要首先调用Preprocessor的config_dataset，选定数据集类型（或名称）、格式（RAW/DTS）、压缩级别（LV0~3，仅对ImageNet有效），这时，Preprocessor应该把具体所要使用的那一个DataLoader的引用（或指针）保存在某个成员变量中（例如current_dataloader_）。在那之后，用户可调用Preprocessor的config_dataloader，这个方法就相当于现在的UnifiedDataLoader的简化版configure方法，里面可以统一配置所有6种具体的dataloader，但是因为前面已经通过config_dataset选定了数据集，所以其实只需配置其中一种。在config_dataloader中，我们可以选择加载模式是partial还是fully（这个选项只对ImageNet有效，对于其他则无效，但具体的DataLoader自身会处理，无需设置特殊逻辑分支），可以设定数据集路径（具体到文件夹，而非具体到文件），可以设定下载与否，可以设定训练集是否shuffle（我们这里强制验证集不shuffle，因为毫无意义）。注意了，这里我们强制要求训练集和验证集同为partial或同为fully。我们并不需要一个config_train_loader和一个config_val_loader，因为训练集和验证集的加载有很多地方可以共享参数。

设定完成后，就为训练集和验证集分配好了缓冲区内存——对于MNIST和CIFAR，显然是固定的FULLY；而对于imagenet，FULLY是分别为训练集和验证集各分配一个单缓冲，PARTIAL是训练集和验证集共享双缓冲——这些都是各个Loader已经安排好的，我想应该不需要修改，只是要用正确的方法调用罢了。必须强调的是，虽然缓冲区内存已经分配好了，但我们**每个epoch可以选择只加载训练集或只加载验证集**。

设定完成后，应该立即检查指定路径、执行下载解压（如有）、读入DTS的文件头、或读入imagenet数据集的摘要（.bin文件）——这些都是整个程序只执行一次的操作，一定要避免每个epoch都执行！

再然后就是config_preprocessor方法，这个方法会设定batch size、world size、SDMP。

这里有一个基本设置顺序的状态机。首先是完全没设置的状态。第一步必须调用config_dataset，然后第二步是config_dataloader，第三步是config_preprocessor，第四步是set_train_transforms和set_val_transforms（两个都调用过才算完成第四步）。全部按顺序设置完成后，才会设置inititalized=true。如果违反了顺序，就要报ValueError，说前一步XXX设置未完成。

这几个方法的签名初步设计如下：

void config_dataset(const std::string& dataset_name, bool dts_format = false, int compression_level = 0);

void config_dataset(DatasetType dataset_type, bool dts_format = false, int compression_level = 0);  // 重载

void config_dataloader(const std::string& dataset_path, int num_load_workers, int num_preproc_workers, bool partial_mode = true, bool shuffle_train = true, bool download = true);

void config_preprocessor(int world_size, int batch_size, int max_resolution, int num_color_channels, int sdmp_factor = 1, bool using_cpvs = false);

void set_train_transforms();  // TODO: 以后会修改，先占个位

void set_val_transforms();  // TODO: 以后会修改，先占个位

这里说一下，sdmp_factor是用于优化训练集预处理的参数，而using_cpvs是用于优化验证集预处理的参数，具体用法不用管，先保存下来，后面再用。我们要求预处理后的图像是正方形，max_resolution是指边长，那么每张预处理后的图片所占字节数就是max_resolution * max_resolution * num_color_channels。而单个缓冲区的大小就是batch_size * max_resolution * max_resolution * num_color_channels。

这里还要说一下，有一个deployment模式，这个模式是调用SampleLoader的，它其实相当于不使用任何数据集，但是把加载的新样本当做val集一样来处理。这个模式要求单线程加载、单线程预处理，并且world size也是1。它会默认把输入的标签都置0。这个模式下需要知道的就只有batch_size、max_resolution、num_color_channels。

void config_deployment_mode(int batch_size, int max_resolution, int num_color_channels);

void set_deployment_transforms();  // TODO: 以后会修改，先占个位

如果用户第一步调用config_dataset，那就是普通模式；如果用户第一步调用config_deployment_mode，那就是deployment模式。两者互斥。如果进入了deployment模式，再调用config_dataset就要报错，反之亦然。

deployment模式也是要先调用config_deployment_mode再调用set_deployment_transforms。



当前的几个DataLoader的实现，在功能、性能、可复现性上都非常令人满意，但测试毕竟只覆盖了少数情况。

我比较担心的是什么呢？一是担心存在重复读取文件头的逻辑。比如每个epoch就读一遍。二是担心当前的DataLoader只支持训练集或验证集两者其一，一旦要求轮流加载两者，就会存在反复释放和申请内存的情况，或者需要重复配置。三是担心不能自由选择加载训练集或验证集、不支持某些epoch跳过训练集或验证集。

我们需要确保FULLY模式下默认会为训练集和验证集都分配内存空间（单缓冲各一个），而PARTIAL模式（仅ImageNet）是训练集和验证集共用双缓冲（这需要检查各个DataLoader的实现）。

我们的框架不需要支持batch_size的变化，因为算法中通常用不到，没有好处，没有意义，而且实现复杂，增加难度，可能引入bug。没有任何算法表明batch size的变化能对训练速度或准确率带来帮助。



另外，Preprocessor需要有一个warmup方法，这个方法就是让loader先跑一个epoch（包括训练集和验证集），Preprocessor迅速地取用完所有已加载的样本（就是完全不执行任何预处理，也不解码，直接取用完样本就结束），使得DataLoader把训练集和验证集都整个加载完成一遍，这样缓存就热了。

Preprocessor还需要有一个test_dataloader方法——测试加载速度及完整性。这个方法就跟现在的test_partial_mode和test_fully_mode的作用相同，完成加载训练集，测试用时、吞吐量、样本总数，然后完成加载验证集，测试用时、吞吐量、样本总数，打印结果。

事实上我认为warmup方法可以复用test_dataloader的实现，不同之处仅仅在于它不打印任何东西。

我们注意到，在之前的测试样例中，loader.begin_epoch(0, is_train)、preproc.run(loader)、loader.end_epoch()这三个关键方法总是连在一起执行的，我们在新版的Preprocessor中，可以把它们合在一起，也可以同时提供这三个接口，允许连续执行或分开执行。这样一来，在缓存友好性上应该不会输给之前的ImageNetLoaderRaw。

事实上我认为最好的办法就是三者合在一起，形成一个train()和val()方法（deployment模式就只允许调用val()方法），而epoch的ID就是内部自动计数，没有调用过train方法的时候，epoch ID就是0，每调用一次train方法，epoch ID就自加1，而调用val()方法不影响epoch ID。更严格地来说，这个epoch只是dataloader认为的epoch，事实上跟用户定义的epoch可能是有出入的，我认为最好换个名称不要叫epoch。

这样修改过之后，测试样例其实就变得更加简洁——测试样例中并不需要直接调用任何的DataLoader，而是调用Preprocessor的配置方法，配置完成后调用它的test_dataloader方法即可。测试样例我觉得最好写在tests/integration/test_dataloader_performance.cpp。用户通过命令输入--dataset、--using_dts、--lv、--path、--load_workers、--preproc_workers、--partial_mode、--shuffle即可轻松设置测试对象。最好添加一个--train_only和--val_only，表示只测试训练集或只测试验证集（两者都为false就表示依次测试训练集和验证集）。