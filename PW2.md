## 内存安排

关于内存，我们再次强调，每个PW内部必须是一次性分配一整块内存，然后划分不同区域来使用，这样就避免了NUMA架构下被分配远端内存。内存的划分是这样的：

**D区（Region D）**：解码区。专用于存放ImageNet样本解码后的数据（包括DTS Loader和RAW Loader，RAW Loader读取的是原始数据集，两者都是JPEG格式），不区分训练集和验证集（它们相当于分时复用D区）。其他数据集完全用不上，可设置为0。D区大小可根据RAW/DTS以及压缩级别LV来硬编码，事实上只要能够容纳解码后最大的样本就行。具体如下。

|                  | 训练集样本解码后最大字节数 | 验证集样本解码后最大字节数 |     建议D区大小     |
| :--------------: | :------------------------: | :------------------------: | :-----------------: |
|   ImageNet RAW   |         95,001,984         |         54,744,690         | 100,663,296（96MB） |
| ImageNet DTS LV0 |         95,001,984         |         54,744,690         | 100,663,296（96MB） |
| ImageNet DTS LV1 |         3,037,200          |         2,174,400          |  4,194,304（4MB）   |
| ImageNet DTS LV2 |          720,000           |          720,000           |  1,048,576（1MB）   |
| ImageNet DTS LV3 |          720,000           |          720,000           |  1,048,576（1MB）   |
|    其他数据集    |            N/A             |            N/A             |   0（不需要解码）   |

这里补充说明一句，我们暂且把Loader的缓存成为I区（输入区），它不属于PW，但是是我们最初的输入来源。数据就是从那里读取的。

我比较建议把判断D区大小的逻辑放在Preprocessor类，这样PW就不需要考虑计算D区的逻辑，也避免每个线程重复计算。

**AB区**：乒乓复用的预处理工作区。从D区解码后（对其他数据集则是从Loader取出后），假如需要1步操作，那么就是从D区直接到输出区（我们称之为O区），但尽管如此，AB区依然存在（我们不想把逻辑弄得太复杂）。假如需要2步操作，那就是从D区到A区再到O区。假如需要3步操作，那就是从D到A再到B再到O。假如需要4步，那就是DABAO。5步，就是DABABO，以此类推。总之，如果是最后一步操作，就输出到O区；如果不是，那就输出到AB区当中的一个（但不能是当前操作的输入来源区）；如果AB区都可用（也就是说是第一步操作且不是最后一步操作），那就输出到A区。再次强调，AB区必定存在，且大小相等。AB区大小的计算逻辑同样是交给Preprocessor。毫无疑问，AB区的大小需要对齐。

**T区（Region T）**：暂存区（Temp）。这个区默认不分配（大小为0），但留着接口备用。我们建议PreprocessOperation类留有一个bool require_temp() const方法，表明这个操作是否需要一个暂存区。如果PW接收到的操作列表里有一个PreprocessOperation对象是需要暂存区的，那么创建的时候，T区的大小就不能为0。使用T区的方法是取决于PreprocessOperation的子类的，基本上就是，从A到B/从B到A/从A到O/从B到O/从I到O的时候，借助T区存放中间产物。但是绝大多数情况应该都是不需要T区的。这个区的大小计算逻辑同样是交给Preprocessor。毫无疑问，T区的大小需要对齐。

**C区（Region C）**：如果using_cpvs为true就创建这个区。CPVS是Cached Preprocessed Validation Set的缩写。这个区是用于暂存预处理后的验证集的，里面存储的就是最终的处理结果。只要使用了CPVS优化，且C区不为空，那么以后的每个epoch的验证环境都只需要从C区读取，不需要重新加载和预处理验证集。这是真正的“一劳永逸”。C区的大小是max_region_c_size。每个Engine（CPU或GPU）对应的验证集样本总数，应该是num_val_samples_per_engine = (num_val_samples + world_size - 1) / world_size，也就是向上取整。那么，max_region_c_samples = (num_val_samples_per_engine + num_workers_per_engine - 1) / num_workers_per_engine，也是向上取整。那么，C区的大小应该是max_region_c_size = max_region_c_samples * aligned_max_val_output_size。这里的aligned_max_val_output_size是对齐后的max_val_output_size，而max_val_output_size = max_val_resolution * max_val_resolution * num_color_channels。再理顺一下C区的使用逻辑，那就是，如果using_cpvs为true，那么第一次加载和处理验证集的时候就会先把处理好的结果放进C区，然后立刻把同一个样本复制到EngineBuffer，并在epoch结束时标记C区已写，此后的每个epoch首先检查是否C区已写，已写的话就直接从C区复制样本到EngineBuffer；如果using_cpvs为false，那就每个epoch都重新解码和预处理验证集，并且每次都是直接输出到EngineBuffer。

**S区（Region S）**：如果sdmp_factor＞1就创建这个区。这个区其实应该分为S1区、S2区、S3区……从1开始编号。S区的个数，取决于sdmp_factor。num_region_s = sdmp_factor - 1。这个区是用来暂存预处理后的训练集的。SDMP就是Single Decode Multiple Preprocess。我们把epoch分为busy epoch和lazy epoch，只有在busy epoch需要加载、解码、预处理训练集，而lazy epoch不需要。毫无疑问，如果sdmp_factor = 1，那就是没有开启SDMP优化，那就是不需要S区，那么每个epoch都是busy epoch。如果sdmp_factor = 2，那就是有一个S1区，每2个epoch当中，有1个busy epoch和1个lazy epoch。如果如果sdmp_factor = 3，那就是有一个S1区和一个S2区，每3个epoch当中，有1个busy epoch和2个lazy epoch，以此类推。在busy epoch，PW会解码一次输入图像放在D区，然后对D区的解码后图像进行sdmp_factor次预处理操作，最后一个放在EngineBuffer，而其他的都放在S区；在lazy epoch，PW需要做的只是从S区复制数据到EngineBuffer，整个epoch只使用同一个S区（比如S1区），不需要做加载、解码、预处理。每个Engine对应的训练集样本总数，应该是num_train_samples_per_engine = (num_train_samples + world_size - 1) / world_size，也就是向上取整。那么，max_region_s_samples = (num_train_samples_per_engine + num_workers_per_engine - 1) / num_workers_per_engine，也是向上取整。那么，S区的大小应该是max_region_s_size = max_region_s_samples * aligned_max_train_output_size。这里的aligned_max_train_output_size是对齐后的max_train_output_size，而max_train_output_size = max_train_resolution * max_train_resolution * num_color_channels。

EngineBuffer我们暂且称之为E区。E区并不是PW的一部分，E区在外面创建，并且num_workers_per_engine个PW共享一个E区。我们可以认为，C区、S区（包括S1、S2、S3等等）、E区都是O区的一种，只不过不同情况下，使用不同的O区。

当然，除了上述内存，我们显然还需要专门的数据结构来存储C区和几个S区的样本标签，或许还需要数据结构来存储洗牌后的读取顺序（洗牌不移动C区和S区的样本，只修改读取顺序，但标签和样本要保持原有对应关系）。

毫无疑问，因为SDMP和CPVS优化的存在，我们的Preprocessor的train和val方法肯定要修改，毕竟现在的train和val都是每次都调用了loader的。现在的实现只适用于sdmp_factor = 1且using_cpvs = false的情形。对于训练集，如果使用了SDMP，显然在lazy epoch就不需要加载和预处理；对于验证集，如果使用了CPVS，显然就只需要加载和预处理一次验证集，随后的所有epoch都不需要加载和预处理。一个epoch是否属于lazy epoch，以及是否需要进行验证集预处理，这些都交由Preprocessor控制，PW只需要在phase开始的时候获取信息即可（这个信息也应该包含在update_parameters的结构体里）。

特别说明，在NUMA架构下，内存分配在本地非常非常重要。在预处理线程首次创建PW并为工作区分配内存时，一定要进行First-Touch，确保该工作区的内存分配在本地。由于我们对Linux GPU云服务器会进行多线程绑核，所以线程重启后，相同ID的线程应该还会对应相同的PW和相同的工作区内存，这样就确保了内存在本地，避免了频繁访问远端内存的开销。这里补充一句，Loader的缓冲区内存是此前已经分配好的，会在其中某一个节点，那么必然有一些PW访问Loader会是远端访问，这个没有办法，也不需要专门处理，我们需要保证的只是：PW工作区内存、预处理线程及其所在的CPU核、相应的EngineBuffer（包含锁页内存）、相应的GPU在同一个节点。由于我们的线程已经绑定到了与GPU亲和的CPU核心，所以只要保证内存的分配位置就行。

## 洗牌

如果用户在Preprocessor设定shuffle = true，那就是要进行洗牌。这种情形下，我们会把shuffle设定传导到Loader，使得Loader给出的样本就是乱序后的样本。但是，由于使用了SDMP，这会导致busy epoch及其后的几个lazy epoch的顺序都一样。那么，在lazy epoch的开始，就需要先对相应的S区的样本进行洗牌。毫无疑问，S区包含的样本总数并非训练集样本总数，但这个程度的洗牌已经足够了，因为后面我们还会把几个PW的输出放到同一个EngineBuffer，然后GPU之间又还会进行Ring-All-Reduce，这可以使得洗牌的范围相当于扩大到整个数据集。洗牌只需要在lazy epoch的train阶段的开头对相应的S区执行一次，随后就只管读取就行。洗牌的范围是这个S区实际拥有的样本总数，这个可能需要一个计数器（毫无疑问，同一个PW的S1、S2、S3区的样本个数肯定是一样的，所以只需要一个计数器就行）。再次强调，洗牌需要保持样本跟标签的对应关系，而且，洗牌只改变读取样本的顺序，不改变样本在S区内部的排列。

## 多线程绑定CPU核心

在NUMA架构的服务器，如果不绑定核心、不设法控制内存的分配位置，难保内存会放在远端，导致延迟很大。在数据集样本数很多的时候，这个延迟造成的时间开销会非常可观。所以，我们的思路是，PW的内存、运行PW预处理功能的线程、GPU、锁页内存必须在同一个节点。这要求我们把预处理线程绑定到与相应GPU亲和的CPU核心，然后分配的PW内存要在这个核心的本地内存。绑定方法应该参照tests/bind/a.cpp，那是我们经过验证的。

这里说一句，多线程绑核的做法，只适用于带NVIDIA GPU的Linux服务器，其他任何情形（比如Windows系统，比如嵌入式的Linux），都不需要。标记这种情形的全局宏是TR_SCENE_GPU_CLOUD = 1，在根目录的CMakeLists.txt有定义。出于测试的目的，我们希望带NVIDIA GPU的Linux服务器也有不绑核的选项（也就是行为跟其他平台一样），所以，我们需要一个宏，叫AUTO_CPU_BINDING，默认开启，但如果不开启的话，那就是不进行绑定。

绑定核心，应该是在Preprocessor类刚完成world_size、num_preproc_workers等设定之后，立刻计算绑定策略。绑定策略的实现就是tests/bind/a.cpp。当然，前提是在带NVIDIA GPU的Linux服务器，且开启了AUTO_CPU_BINDING的条件下。

我们强调，“可见的GPU数”与world_size不是一个概念。我们要求用户从可见GPU中选取一个子集来使用，这个子集的GPU数量必须是2的幂或0（0就是使用CPU训练）——那么，选取的GPU数不为0的时候，world_size就是GPU数；选取的GPU数为0（或压根就没有可见GPU）的情况下，world_size就是1。

我们已经在Preprocessor里面提供了world_size参数的输入，但这显然还不够，我们还需要添加探测GPU总数的逻辑以及让用户指定使用的GPU ID的逻辑。这需要使用TR_USE_CUDA宏把它括起来。这是涉及到CUDA Runtime库的。

绑定只执行一次，就是在预处理多线程首次展开时。绑定时一定要有debug信息说明哪个线程绑定到了哪个核心。

## PreprocessOperation

关于预处理的操作，我们设置一个基类，叫PreprocessOperation（简称PO）。但命名空间似乎应该叫tr::transforms。它的子类有：ColorJitter、RandomErasing、RandomGrayscale、GaussianBlur、RandomRotation、RandomAutocontrast、RandomCrop、Resize、Pad、CenterCrop、GaussianNoise、DoNothing、RandomScale、RandomHorizontalFlip、RandomResizedCrop、FastRandomResizedCrop、CalculateCrc32。我必须强调，这些类我们并不是一下子就都要实现。只是最好留个接口，标明TODO。只要有第一个验证通过了，其他的参照同样的模式都很快就能设计完成。我们规划要在早起实现的，就是DoNothing、Resize、CenterCrop、RandomHorizontalFlip、RandomResizedCrop、FastRandomResizedCrop这几个。

我们尽量维持用户自己定义的顺序，但是对于RandomHorizontalFlip如果有出现，我们自动把它放在最后。RandomHorizontalFlip是一个特例，它不一定需要操作，有50%概率是原封不动的。如果不需要操作，我们最好把这个内存复制也省掉。把它放在最后，目的就是，有一半的概率可以把倒数第二个操作的结果直接导到输出区，而另外一半的概率则是先导到另一个工作区然后翻转再导到输出区。另外，对于ImageNet，我们要求第一个必须是Crop或Resize，因为如果图像尺寸没有确定的话，后续很多操作都很难开展。PreprocessOperation必须要有方法叫is_crop()、is_resize()，还要有个方法叫is_random_operation()，表示这个操作会引入随机性——对于验证集，如果所有操作都没有引入随机性，那就可以实施CPVS优化。is_crop()和is_resize()还有一个用处，那就是适配我们的渐进式分辨率。渐进式分辨率的train_crop_output_size、train_resize_output_size、val_crop_output_size、val_resize_output_size数组，里面的值就是赋给Crop操作和Resize操作的。DoNothing相当于一个占位符，它在有其他操作的时候，可以直接省略，当做不存在，但如果没其他操作，就表示直接从I区复制到O区的操作。注意了，因为DoNothing不属于Crop或Resize，所以它显然不能单独用在ImageNet的预处理上。然后就是，我们必须有局部解码和完全解码两个方法，这两个方法是供Crop和Resize调用的，它们会确定应该解码哪些区域。事实上，只有Crop和Resize可能会用到D区——只要是imagenet，且它们是放在第一个的操作，那它们就会首先调用解码。

至于应该调用局部解码还是完全解码，规则如下：Resize在首位时，必定调用完全解码；CenterCrop、RandomCrop在首位时，必定调用局部解码；RandomResizedCrop在首位时，如果sdmp_factor=1就调用局部解码，如果sdmp_factor > 1就调用完全解码；FastRandomResizedCrop在首位时，必定调用局部解码。不在首位的任何操作，都不需要调用解码。是否调用局部解码以及局部解码哪个部分，这个是在相应的PO里面可以计算得出的。

有了这样的限制，我们可以确保其他操作所需要处理的都必定是解码后的、正方形的图像，复杂度降低很多。

这里说一下，RandomErasing虽然是在Preprocessor中定义，但实际操作是在GPU上完成的，它不会被分配给PW。

我们的PreprocessOperation的子类不包括ToTensor和Normalize。ToTensor操作在GPU上完成，而且我们的框架强制执行这个操作。Normalize我认为可以独立出来，做一个设定，就是Preprocessor有一个set_normalization_mean方法和一个set_normalization_std方法，接受1个float或3个float，然后把数值注册到全局注册表，等待后续GPU处理，仅此而已。

现在我们需要定义它的一个很重要的新行为，那就是设置预处理操作。它应该有一个set_train_transforms方法，和一个set_val_transforms方法。它们都可接受若干个tr::transforms::PreprocessOperation类的子类的对象。或许可以考虑用可变参模板来实现：template<class... Args>、void broadcast(Args&&... args)。

我们要求先设定workers的数量，再设定预处理操作类型。这样我们就可以在设定预处理操作类型时，把它们“配发”（复制）给每一个worker。预处理操作的类对象都是不怎么占内存的，每个worker各拥有两套（train和val）也是没问题的。这里要注意，“配发”是有顺序的，并且是被检查过、调整过的顺序（RandomHorizontalFlip后置）。对于所有的PreprocessOperation的子类，我觉得都必须有一个execute方法，就是进行实际的处理操作。

补充说明一下，即使worker的train和val工具集里都有相同的操作，它们也是两个不同的对象，不会共用，以免影响统计。

## PreprocessOperation补充

PO相当于工具，每个PW有两套PO，一套用于训练集预处理，一套用于验证集预处理。这两套PO都是在初始化时从Preprocessor类复制过来的，保持原有顺序。我们要求PO是简洁、轻巧的，不占太多内存，以便每个PW都能复制一套PO。PO主要是拥有一套预处理算法以及相关参数，这个预处理算法很可能通过Simd库进行实现，目前我们希望所有的预处理操作都是从int8_t到int8_t的。

PO内部尽量不要再展开多线程，因为调用PO已经是多线程调用了，每个时刻每个线程都运作一个PO。

PO需要有bool introduce_randomness() const方法，以反馈这个预处理操作是否引入随机性。

还要有一个std::string name()方法，返回名称。

要有一个bool require_temp()方法，表示需要借助额外的T区来进行预处理（一般都不需要）。

注意，对于训练集和验证集，所执行的预处理通常是不一样的，但是，我们强制要求它们共用PW，也共用Workshop（D、A、B、T区）。因此，Preprocessor显然需要能够区分当前是出于训练阶段还是验证阶段。PW会控制工作流，它会确定第一个操作是从哪到哪，第二个操作是从哪到哪，把输入输出指针定好。所有的PreprocessOperation子类，它们都不会分配自己的工作区，它们只是持有参数，以及定义运算而已。每个对象应该都很小，用户把它们传给Preprocessor，而Preprocessor会把它们复制给每一个PW。PreprocessOperation子类应该具有一个void execute(int8_t* input_ptr, int32_t input_bytes, int32_t input_width, int32_t input_height, int8_t* output_ptr, int32_t& output_bytes, int32_t& output_width, int32_t& output_height)方法（API仅供参考），就是实际的计算。

简单来说，预处理过程就是PW根据阶段（train/val），按顺序依次调用各个PO的execute方法，并妥善指挥输出。

我们认为，JPEG解码（包括局部解码和完整解码两种）不应该算作PO的子类，而是应该实现为PW的方法。原因在于，这不是一个需要用户配置的操作，而且它的位置、设定、用法相对固定。若是实现为PO的子类反而增加复杂度。

还需要一个is_resize()和is_crop()方法，表明预处理操作的类型。毕竟我们要求对于ImageNet，第一个操作必须是Resize或Crop。

我们认为，对于Resize和Crop类的PO，它需要有一个bool using_partial_decode()方法和一个get_decode_region方法，来判断是否需要局部解码以及具体要解码哪个区域。后者可以是由该PO随机生成的。对于ImageNet（无论RAW还是DTS），首先就要根据第一个PO的要求来进行解码（视情况选择局部解码或完整解码），放到D区，然后再把解码后的数据的指针、字节数、长宽等传递给那第一个PO的execute方法。这里需要提醒的是，对于JPEG局部解码，是需要对齐到MCU的。

由于PO本质上只需要从指定地点取出数据，处理完后放到指定地点，都是点到点的，所以其实实现相对容易，没有复杂的逻辑。

RandomHorizontalFlip的实现其实也不复杂，它同样是点对点，但它需要给出一个yes/no的答案，比如提供一个bool should_flip()方法，以便让上层的PW判断这一回合是走有Flip的流程还是无Flip的流程——两个流程的指针传递是不一样的，不需要Flip的时候，少一次操作，也少一次AB区的使用。

此外还需要注意的就是PO的排序。因为PO是用户在Preprocessor的set_train_transforms和set_val_transforms方法里设定的，设定之后Preprocessor需要负责检查它们的排序。对于ImageNet，第一个PO必须是Resize或Crop；如果有RandomHorizontalFlip，则自动把它放在最后，无需报错和提示；其他就按照原有的顺序排列。

此外还需要注意，RandomResizedCrop是可以放在第二位的，比如在MNIST数据集。其他Resize和Crop也是。

## PW运作逻辑补充

我们认为，在线程重启后，更新PW参数后，就只需执行一个work()方法，就把所有的操作都按顺序依次完成。根据phase选择不同的PO组。对于ImageNet，获取样本之后的第一个操作必定是解码，可能是完整解码也可能是局部解码，随后就是依次执行PO组内的所有操作，在AB区乒乓，最后输出到O区。对于训练集，O区包括S区（如有）和E区；对于验证集，O区包括C区（如有）和E区。对于训练集的busy epoch，PW进行一次解码和sdmp_factor次预处理，依次放到S1...Sn区和E区，E区最后。对于训练集的lazy epoch，PW从其中一个S区（具体是哪个要取决于epoch ID）逐个样本地复制数据到E区。对于验证集的首次预处理，会进行一次解码和预处理，放在C区然后复制到E区；对于其后的轮次，都是直接从C区逐个样本地复制到E区。

如果有shuffle，则需要在lazy epoch之初进行一次相应S区的洗牌。

PW的work方法关键就在于要根据当前阶段（train/val）和epoch ID（busy/lazy，首次/非首次，训练集验证集epoch计数区分开）来控制输出位置，以及要根据RandomHorizontalFlip的随机情况来控制最后一步。

## EngineBuffer

我们需要定义一个EngineBuffer类。它有CpuEngineBuffer、CudaEngineBuffer、MusaEngineBuffer和EngineBufferEmulator这4个子类。我们简单来说，CpuEngineBuffer管理着2个一般内存的缓冲区；CudaEngineBuffer管理着2个CUDA锁页内存缓冲区的BatchBuffer和2个CUDA显存的缓冲区；MusaEngineBuffer管理着2个MUSA锁页内存缓冲区和2个MUSA显存的缓冲区；而EngineBufferEmulator其实是用来模拟CudaEngineBuffer和MusaEngineBuffer（因为这两个类暂不实现），它管理着4个一般内存缓冲区。这个设计的意图非常明显，那就是我们采取双缓冲+异步传输的策略。从锁页内存到显存的复制，我们只复制“最短有效长度”，具体来说就是local_batch_size * 4 + current_resolution * current_resolution * num_color_channels * actual_num_samples，这样可以极大地节省带宽。除了EngineBufferEmulator以外，其他3个子类显然都应该由相应的Engine来创建。
EngineBuffer最重要也最关键的一个功能，就是双缓冲切换。如果是4个缓冲区的，那就是有两个是输入缓冲区、两个是输出缓冲区；如果是2个缓冲区的，那显然这两个既是输入缓冲区、也是输出缓冲区。当一个输入缓冲区被写满，那就立刻触发异步传输，然后检查另一个输入缓冲区是否可写；如果可写就向另一个缓冲区写入，如果不可写就只能等待。EngineBuffer和EngineBufferEmulator类应该放在src/data，而其他3个子类应该放在src/device。

EngineBuffer其实还应该预留一个区，就是返回区，用于从GPU向CPU返回数据。

EngineBuffer主要是多线程写入的逻辑和切换双Buffer的逻辑。

首先再次重申，我们的EngineBuffer的数量等于world_size，每个EngineBuffer对应的PW个数是num_workers_per_engine。

PW（或者说预处理线程）与EngineBuffer的对应关系是跨步分配的，第i个PW（或线程ID为i的预处理线程，从0开始编号）对应的EngineBuffer编号为engine_id = i % world_size。跨步分配的作用是保证每个Engine最终分配到的样本数差值最大不超过1。

不管是训练集还是验证集，显然都是放在同样的EngineBuffer，只不过是分时复用而已。

我们初步考虑，EngineBuffer里面的图像数据不需要对齐也不应该对齐，否则后续就没法轻易转化为Tensor。

EngineBuffer的双缓冲之一的大小应该是single_buffer_size = local_batch_size * 4 + max(max_train_output_size, max_val_output_size) * local_batch_size。前面的local_batch_size * 4指的是int32_t类型的标签。

接下来是最重要的地方：多线程写入。我们并未要求num_workers_per_engine必须是2的幂，也并未要求local_batch_size必须是num_workers_per_engine的整数倍。所以，每个PW其实需要计算它在这个batch里面应该往EngineBuffer的哪些位置写入数据，以及应该写入多少个样本的数据。举个例子来说，如果local_batch_size是8而num_workers_per_engine是3，那么第一个PW就应该写0,3,6号位，而第二个PW写1,4,7，第三个PW写2,6就完成；再下一个batch，第三个PW写0,3,6，第一个PW写1,4,7，第二个PW写2,6。这其实是可以固定分配、完全不冲突的。这个计算非常重要，一定要把公式写好。我们在PW划分到EngineBuffer后，每个PW可以领到一个在那个EngineBuffer里面的编号，比如说叫pw_id_in_engine（简称PID），对于同一个EngineBuffer的PW来说，PID是连续的，并且，线程ID较小的线程领取到的必须是较小的PID。如果PID用j表示，那么需要写入的位置可能应该写作position = (n * num_workers_per_engine + j) % local_batch_size。其中n就是该PW写入的第n个样本。每个batch当然有一个累积样本总数范围，如果计算出来的n * num_workers_per_engine + j超出了这个范围，那显然就是当前Buffer装不下，那就是视为这个PW在当前Buffer的任务已全部完成。写入时还有一个需要注意的就是，每个样本的字节数以及写入的偏移量，是要取决于当前是训练集还是验证集的，也要取决于当前的current_resolution。这些都是在phase之初就配置好的。这意味着EngineBuffer也需要一个update_parameters方法，在每个phase开始之前调用。

关于传输唯一需要注意的是，我们在所有PW都完成写入后，需要触发异步传输，然后切换到双缓冲当中的另一个缓冲。如果在最后一个batch，可能有的PW已经用完了样本，那么就直接结束，不需要写入。为了减轻复杂度，我们现在的EngineBuffer只是模拟，所谓的异步传输就是不做任何操作，只发一个debug消息说传输完成。

我们这里的EngineBuffer，最好还是实现为一个抽象类，它的子类包括CpuEngineBuffer、CudaEngineBuffer、MusaEngineBuffer和EngineBufferEmulator。主要差别是在创建缓冲区内存的方法以及异步传输的实现方法上。前面三个我们暂且不管（因为肯定需要锁页内存、显存或者内存上的4个buffer，来实现传输，很复杂），我们先只实现EngineBufferEmulator，那是双buffer，并且没有实质上的传输，它的传输只是输出一个debug信息而已，就说缓冲器A或B已传输，当前resolution是XXX，有效样本数是XXX，传输的总字节数是XXX。最好它能有一个总数统计，好让我们知道它最终一共传输了多少个样本。

设计EngineBuffer的时候，一定要特别注意NUMA架构下的多线程写入，要避免同步超时和死锁。理论上我们静态分配写入的地址是可以避免竞争的，但关于“所有线程写入完成”以及每个线程写入完成之后的等待（等待其他线程写入当前buffer完成以及等待下一个buffer变为可写），需要特别注意。我们必须强调，当一个线程把当前buffer（或者说当前batch）写入完成后，如果其他线程没有写入完成，那么即使下一个buffer为可写状态，也不可以立刻跳过去写入。我们需要确保每一个时刻只有一个buffer正在被写入，而不能一些线程在写Buffer A、另一些线程在同时写Buffer B。

## 测试样例

我们之前其实开发Preprocessor失败过一次，就是出现了bug查不出来，而且性能损失极大却不知道瓶颈出在哪。一口吃不成大胖子，我们必须步步为营地进行开发测试。

我这里只说第一步、第二步和第三步。第一步就是开发PO类和Resize和CenterCrop这两个子类。这是可以单独用图片来验证效果的。测试样例就是输入图片，解码后传递给这两个PO之一，让它们输出处理后的图片，保存。再次强调，PO的子类数量很多很多，我们先重点完成Resize和CenterCrop，因为它们一个对应完整解码一个对应局部解码。千万不要一下子试图实现所有的PO。

第二步就是开发简单的PW。需要实现局部解码、全局解码和文件头读取方法。注意，局部解码是需要输入参数的，读取文件头后，应该由相应的PO判断进行何种解码，然后再把解码区域传递给局部解码方法，解码到D区。这个PW需要包含完整的所有区的实现，我们也需要在测试样例中运行C区和S区的写入逻辑。同时，EngineBufferEmulator也要实现，用以配合测试。但是只要实现最简单的功能即可。

第三步就是实现PW的绑核，这一步需要集成到Preprocessor。Preprocessor显然需要经过一些修改。但目前的train()和val()的逻辑都是正确的，并且是经过了CRC验证的。

这里提醒一句，如果没有特别情况，尽量不要去修改任何一个DataLoader，因为那会引发未知的bug或造成性能衰退，破坏此前辛苦测试的成果。

大概就是这样。