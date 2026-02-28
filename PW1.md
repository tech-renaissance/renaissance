

# 【三、新版基本设计方案】

## 现状

技术觉醒框架的开发到现在为止，已经取得了非常突出的进展。我们已经做出了非常强大的数据集加载器，配合我们自定义的二进制格式DTS，可以实现ImageNet的超快速度加载，我们的PARTIAL模式双缓冲占用内存仅2GB。ImageNet的RAW加载器加载整个数据集用时仅 **7.036 s**，ImageNet的DTS加载器加载整个数据集用时仅 **4.706 s**，如果加载的是LV3压缩的数据集，用时可进一步缩减到 **1.806 s**。我们的加载已经通过了CRC验证，确保数据集完整；而且还通过了随机可复现性测试，相同的随机数种子可以得到相同的加载结果。此外，四大数据集（ImageNet、MNIST、CIFAR-10、CIFAR-100）的原始格式和DTS格式加载均已成功实现，速度令人满意，可以很肯定地说加载已经不再是我们项目的性能瓶颈。**如果没什么特别的事，尽量不要去改数据加载器类的实现，否则重新验证和debug会带来巨量的时间开销。**

现在问题是什么呢？是预处理。更准确地说，是作为预处理第一步的解码的速度。我们已经正确地使用了libjpeg-turbo库进行JPEG解码，速度其实与官方提供的典型值相当（折算成输入吞吐量的话是1GB/s的量级）。但这还是太慢了，对于一个力求挑战PyTorch性能上限的深度学习框架来说，如果预处理速度被卡死在解码速度的1GB/s的话，那么每个epoch的用时必定超过2.5min——而我们的目标是冲进50s以内。

## 突破解码性能瓶颈的三大方案

我已经带来了三个决定性的、必定有效、而且效果稳定的解决方案。

我的第一个方案，叫**DTS压缩**。我们已经说过，DTS格式的ImageNet包括了LV0~LV3这4个层级。LV0是ImageNet原图，完全没有字节修改；LV1是仅缩放；LV2是缩放+裁剪；LV3是缩放+裁剪+降低JPEG质量到80。你只需要知道，读取LV0~LV3可以使用完全一样的加载器和完全一样的配置，而且LV3是尺寸相对规范的图片，LV3的大小差不多是LV0的**三分之一**。LV3并没有因为压缩比更高而需要更多的解压开销，因为它同样是JPEG压缩，而非zip压缩。我们在LV3的规范化设置上做了大量的研究，可以确保这个压缩策略的科学性，使得压缩给训练精度带来的影响微乎其微。

我的第二个方案，叫**局部解码**。libjpeg-turbo提供了一种被称为Partial Decoding（局部解码）的特性，支持只解码图像的某一部分，而不是整张图解码。利用TurboJPEG API（ tjDecompressToYUV() 、 tjDecompressToYUVPlanes()  等）先拿到原始 YUV 数据；然后通过  jpeg_skip_scanlines()  跳过不需要的行， jpeg_crop_scanline()  把左右多余列也跳掉，这样真正送进 IDCT 的 MCU 行/列就只剩下目标区域。不过，这个受限于MCU（Minimum Coded Unit），通常是8×8像素。RandomResizedCrop会随机裁剪0.08到1.00的面积，如果提前知道要裁剪什么区域，然后就只解码这个区域（或略大于这个区域），用时会明显小于解码整张图。这里说一下，我们做过测试，在解码区域较小的情况下，局部解码是有明显提速效果的，但是即使在非常非常理想的情况下，这个提速也只能勉强达到2×，而多数实际应用情景当中，由于种种因素的影响，它可能只有40%左右的提速。如果解码的用时太长，那么光是用局部解码的策略，未必能达到每个epoch用时50s以内的目标。关于基本的RandomResizedCrop算法实现，参考tests/simd/baseline.cpp；关于完整解码和局部解码的实现，参考tests/simd/fast_crop.cpp；关于基本的Resize算法的实现，参考tests/simd/fast_resize.cpp。

我的第三个方案，叫**单解码多预处理**（Single Decode Multiple Preprocess, **SDMP**）。我们注意到一个事实，那就是图像并不是每个epoch都需要重新解码。解码是不具备随机性的行为，它的结果非常固定。但是，如果我们存储解码后的字节流，那又会很大（~700GB），超出绝大多数服务器的内存上限。使用这样的策略的框架就失去了普适性和弹性，对硬件提出过高的要求，最终会被大众所抛弃。如果是在硬盘存储这个超大的解码后的文件，那么占用空间很大不说，还给加载器带来很大负担，这个大小的二进制文件不可能缓存得下，那么出去缓存的加速后，如果每个epoch用2~3GB/s的速度去从硬盘读取，只会更慢。最合理的办法，就是缓存预处理后的小图像。对于ImageNet，这个图像的尺寸通常是224×224，比原图小很多。训练集是179.6GB，而验证集是7.0GB——这是一个比较合适的大小，是服务器的内存可能装得下的。我们解码一次，就对解码后的图像进行1~X次的预处理，得到1~X张小图像。第一张显然是直接送往下一步处理，但其余的就应该缓存起来。有了SDMP，我们第一个epoch已经生成了小图像（输入样本），那么第二个epoch就可以不再加载、不再解码、不再预处理，而是直接使用缓存的数据。所需要的时间仅仅是内存复制的时间而已，速度基本只受限于下游消费的速度。这样一来，虽然前面1个epoch需要负担多几次预处理，但后面的X-1个epoch就可以**完全免除加载、解码、预处理**。事实上，在PyTorch训练ResNet-50的时候，预处理带来的内存开销就超过300GB，我们的框架动用200GB用于缓存预处理数据是非常合理的选择。这个方案很大程度上攻克了解码性能瓶颈，使得我们不需要依赖GPU解码。

这三个方案的实现几乎没有任何难度（而且第一个方案我们已经实现了），其收益也几乎没有任何疑问。

第二个方案跟第三个方案某种程度上讲是互斥的。原因在于，SDMP方案的特点在于复用已解码的图像，但如果配合了RandomResizedCrop，你就会要么丧失速度、要么丧失随机性。如果要求保持随机性，那么多次随机之下，就会使得需要解码的部分变大，大到跟原图相当的地步，这种情形下，局部解码的加速效果就几乎彻底丧失了；如果要求速度，那么你每次都是预处理同一个Crop区域，就丧失了随机性，事实上就影响了数据增强的效果，必定损失精度。整体上讲，局部解码的耗时跟解码的面积成正相关，粗略估计大约可直接节省一半用时。而SDMP，则是必须有一次完整解码，这一次的用时是很长的，但后面一个或多个epoch的用时就会缩减到跟下游用时相当。假如解码和预处理用时是t1，而下游用时是t2，那么使用局部解码之后，用时大概是t=max(0.5t1, t2)；而使用SDMP的话，用时大概是t=(max(t1, t2) + (X-1)*t2)/X——公式看上去复杂，但解释起来就是“把CPU比GPU多出来的用时均摊”。SDMP比较适用于CPU解码特别慢、跟GPU差距很大的情况；而局部解码适用于CPU跟GPU差距不大、希望达到极致性能的情况。SDMP如果要胜过局部解码，除了CPU和GPU差距要很大之外，还要X≥3，也就是内存可能需要≥400GB。

关于如何将第二个方案和第三个方案有机结合，进一步提高速度并且兼顾随机性，就要用到我最新发明的FastRandomResizedCrop算法。

## FastRandomResizedCrop

RandomResizedCrop通常选取面积范围为0.08~1.00。这个分布不是均匀分布，而是对数分布，也就是先对两个边界值求自然对数，在其范围内取均匀分布随机数，再然后取自然指数。这样的结果就是，更倾向于Crop出更小的图形。

在我们的SDMP/局部解码方案下，RandomResizedCrop的行为是：如果sdmp_factor=1（即不使用SDMP），那么解码的时候就是完全的局部解码，只解码要Crop的那部分；如果sdmp_factor>1，那么就解码整张图片，然后对其进行若干次Crop。两种优化没法用在一起。

那接下来是我们的FastRandomResizedCrop。它是局部解码与SDMP的有机结合。如果sdmp_factor=1，那么FastRandomResizedCrop的行为与RandomResizedCrop一致，就是只解码需要Crop的那个局部。特别之处就在于sdmp_factor＞1的情况。首先，当用户输入面积范围[a,b]时，计算边界值的三次方根c=sqrt3(a)、d=sqrt3(b)。对于sdmp_factor=2，你先在[c,d]之间取均匀分布随机数p，然后取p²，这就是要局部解码的面积比例。在解码出来的区域里面，再在[c,d]之间取均匀分布随机数q，以q为比例，对解码出来的区域进行两次Crop，分别完成后面的预处理流程，保存到缓冲区或传送到输出。对于sdmp_factor≥4，你先在[c,d]之间取均匀分布随机数p，这就是要局部解码的面积比例（而不是p²）。在解码出来的区域里面，再在[c,d]之间取均匀分布随机数q，以q²为比例，对解码出来的区域进行多次Crop，分别完成后面的预处理流程，保存到缓冲区或传送到输出。而对于sdmp_factor=3的情形，我们设置一个宏，叫CROP_SCHEME_2，它为true的时候，sdmp_factor=3的情形采取sdmp_factor=2的策略，它为false的时候，sdmp_factor=3的情形采取sdmp_factor=4的策略。这个方案就是执行两步随机，先确定解码窗口，再确定随机Crop窗口。唯一需要注意的是长宽比，我认为长宽比应该只随机一次，以免增加Crop失败的概率。这个策略在sdmp_factor＞1的时候，平均速度必定比RandomResizedCrop快，并且在sdmp_factor = 2或3的时候优势最明显。在提高速度的同时，不会明显丧失随机性。我们必须指出，通过这种策略Crop出来的面积分布是三次方根分布（p²q或pq²），与对数分布虽然有微小的不同，但差距非常小，几乎可以忽略不计，三次方根分布也是倾向于Crop出较小的图形，但整体上比对数分布略大一点。如果我们使用了FastRandomResizedCrop来利用SDMP和局部解码优化，预计可以在第一个epoch解码时间上打七折，然后随后的1~2个epoch又能直接免除加载、解码、预处理，这样带来的加速幅度将是非常可观的。

SDMP优化，针对的是JPEG输入，并且需要足够大的内存。

## CPVS优化

SDMP是专属于训练集的优化，而针对验证集的优化是**验证集预处理缓存**（Cached Preprocessed Validation Set, **CPVS**）。把预处理后的验证集完整地保存在计算机内存中，每个epoch重复利用，这样就节省了解码和预处理验证集的时间。CPVS优化的要求是，验证集的预处理操作里面没有引入任何的随机性（看PreprocessOperation类的introduce_randomness()方法）。这个预处理和缓存只在第一次调用时进行。需要注意的是，我们的框架是允许一些epoch跳过验证的（比如隔几个epoch验证一次，或者只在最后几个epoch验证），这样就把验证的时间节省了下来。我们不建议缓存到GPU显存中，因为GPU显存一般较小，而且用于深度学习，非常宝贵，不应该被验证这种次要操作挤占过多显存。

## 新形势下的新架构

以前说过，我们的框架分为**DP系统**（Data Preparation System）和**DL系统**（Deep Learning System）两大部分。DP系统，做的就是数据准备的工作，它就是把数据集加载、图像预处理的工作完成，最后得到的，是处理后的正方形小图像。这些小图像尚未进行ToTensor处理，原因有二：一是这个转换工作和后面的归一化工作完全可以由GPU代为处理，减轻CPU负担；二是Host-to-Device传输本身就有可能是一个瓶颈，如果转换成了FP32或BF16再来传输，那么带宽限制下，用时将会成倍地增加。DL系统，主要就是模型、训练器和特征图。我必须指出，DL System下面管辖着若干个DL Engine（类名就叫Engine即可）。如果使用CPU进行模型的训练和推理，那么Engine就是CPU；如果使用GPU进行模型的训练和推理，那么每个GPU就是一个Engine。Engine可以是0个或1个CPU（0个代表使用GPU），也可以是U个GPU——这里的U只能是0、1、2、4、8、16这几个数之一。我们非常强调，我们的深度学习框架当前只考虑单机训练，而单机上能挂载的GPU一般最多就是16个。DL Engine的设计还有一个作用，就是在Deployment模式下允许Ensemble推理。Deployment模式下只允许有一个Engine，但这个Engine既可以部署一个Model，也可以部署一个Ensemble。部署Ensemble的时候，就相当于相同的一份输入被多个模型共用，然后它们的输出进行平均或投票。这个是后面的事了。

我们的DP系统其实主要就包括DataLoader和Preprocessor，其中，用户不需要亲自对接DataLoader，一切通过Preprocessor来设置。这样一来，从用户的角度，DP系统几乎就等于只有Preprocessor。我们完全没必要再单独封装一个DP系统类（多一层转发的话，大概率会降低运行速度）。但DeepLearningSystem类或许有必要，因为深度学习算法里有太多需要各模块配合与互动的地方，有一个类负责管理很重要。模型、训练器、特征图等显然属于DL System。

我们在多GPU的场景下，需要区分每个GPU的batch size和总的batch size。这里我们把每个GPU的batch size称为local_batch_size，而总的batch size称为global_batch_size。显然，global_batch_size = world_size * local_batch_size。当使用的是CPU时，local_batch_size指的是CPU上的batch size，又因为CPU上训练的world_size为1，所以使用CPU的情形下global_batch_size = local_batch_size。
在用户通过config_preprocessor设定local_batch_size时（现在好像写的是batch_size，我比较建议更名为local_batch_size，比较具体、明确），我们要求local_batch_size≥1，仅此而已。

当然，我们还需要num_global_batches_train和num_global_batches_val。这是在确认数据集后，根据总样本数，计算出来的global batch的数量。这就是我们在一个epoch中会向GPU发送的批次数。
我们这里强调，global_batch_size、world_size或local_batch_size这三个参数一经设定，就要立刻注册到全局注册表，叫fixed_global_batch_size、fixed_world_size和fixed_local_batch_size，我们不允许再动态变更。后面可以查询，或可以幂等赋值（赋值为与之相同的值不报错，但也不做任何事情），但绝对不能再修改，修改会报ValueError。

Preprocessor在config_dataloader中设定了num_preproc_workers，又在config_preprocessor中设定了world_size，我们在设定world_size设定时，必须确保num_preproc_workers能被world_size整除，否则就报ValueError。我们设置num_preproc_workers_per_engine = num_preproc_workers / world_size。预处理线程应该根据其ID分组，比如，第0号线程到第num_preproc_workers_per_engine - 1号线程总是对应第一个GPU（或CPU）。这会给我们后面的buffer划分带来一定的便利。我们这里再次强调静态分配原则，这种对应关系应该固定下来，而不应该由预处理完成的先后顺序决定，否则多线程竞争带来的延时甚至超时损害是无法估计的，尤其是在NUMA架构下。预处理线程（以及PW）是跨步分配，假定预处理线程或PW编号/ID为i（从0开始编号），那么它对应的GPU的编号是i % world_size。

## 全局注册表GlobalRegistry

这里有一个很关键很关键的设计，叫全局注册表GlobalRegistry。这是一个在base板块的全局单例类，这个类全局可见，它里面保存着全局的参数设定。

成员变量分为两大类：一次性设定后在整个程序运行期间固定不变的，我们称之为固定型，前缀“fixed”；每个阶段的间歇可以改变的值，我们称之为可变性，前缀“alterable”。这些都设为原子类型。

所谓阶段，就是训练阶段（train）或验证阶段（val），通常一个epoch会分为这两个阶段。这些阶段运行期间，很可能会使用多线程并发，所以安全性特别重要。为了避免冲突，我们要尽量避免在多线程展开的时候修改GlobalRegistry的成员变量。

首先，fixed变量，它的初始值应该是一个非法值。我们只允许把它从非法值改为合法值一次。此后就只允许幂等赋值——可以赋值为相同的值（这种情况下不报错也不报任何warning），但是如果试图再次修改就要报ValueError（详见docs/logger_exception.md）。此外，我们需要有一个initialize方法，它会一次性检查所有fixed变量的合法性，换句话说就是检查是不是全都有被赋值。initialize结束后，整个GlobalRegistry就会被设定为bool initialized_ = true，一旦设定为true，再也不会变为false。一旦看到initialized标志，就意味着任何情形下，fixed的变量不再接受任何更改（这个时候即使幂等赋值也要报warning）。

然后就是alterable。GlobalRegistry有个原子的is_training标志位和is_training()方法，也有个原子的is_validating标志位和is_validating()方法。

GlobalRegistry有个begin_train()方法和end_train()方法，以及begin_val()方法和end_val()方法。都是供其他类调用的。此外，它要维护两个原子计数器train_counter和val_counter，记录有多少个对象调用了begin_train()方法和end_train()，以及begin_val()和end_val()。方法很简单，初始值是0，有一个对象调用了begin_train()，train_counter就加一，有一个对象调用了end_train()，train_counter就减一。如果减到负数了就报ValueError。如果train_counter和val_counter都是0，那么is_busy标志位就是false；如果train_counter_和val_counter有一个大于0，那么is_busy标志位就是true。

当is_busy的时候，表明有的框架部件已经进入训练或验证的状态，这时极有可能展开了多线程，这时全局注册表里的所有的注册变量都不允许修改。换句话说，alterable型变量虽然允许修改，但也只能在is_busy为false的时候修改。
每个变量除了有一个专属的getter方法和一个专属的setter方法外（这些方法可以省略掉fixed和alterable前缀），如果是int、float这些，通常还支持字符串名称查找。
这里说一下initialize()方法，这个方法只有第一次被调用时有效果，被调用过一次之后如果再调用第二次就什么也不干了。然后就是，这个方法可以用户手动调用，也可以自动调用——在第一次调用begin_train()或begin_val()时，如果initialized_为false，就自动调用initialize()。目前来说initialize()的主要作用就是检查所有fixed变量的合法性。

GlobalRegistry实现在base目录下，作为最基础的base模块之一。

现在全局注册表类已经基本实现，但还有一些小问题，除了注册的参数还不够多以外，还有就是current_resolution其实应该区分为current_resolution_train和current_resolution_val。

## PW创建

现在说一下**PreprocessWorker**（**以下简称PW**）及相关类的设计。

首先，PW是一个类，它并不直接等同于线程。我们在配置Preprocessor时，会设定preproc的数量（num_preproc_workers），那就是预处理线程数。预处理线程数在整个程序运行期间是固定的，它们可能会join，但重新展开后总是同样多的数目。

这个类我们可以单独地设计和调试，但是最后是交由Preprocessor类来管理的。也就是，Preprocessor持有PW对象的指针，数量等于num_preproc_workers。但是，并非在一开始就直接创建。我们要求预处理线程展开后，各线程自行检查和创建自己对应的PW。

这里说一下，PW一经创建，在整个程序运行过程中就是**保持存在**的（哪怕预处理线程结束），只有在Preprocessor析构的时候才销毁。预处理线程与PW一一对应，如果预处理线程结束后重启，相同线程ID的线程依然对应相同的PW。

简单来说，就是我们希望每个线程拥有自己独占的、其他线程不能访问的PW对象，以此避免冲突。当然，预处理的多线程展开前或结束后，主线程可以直接管理和配置所有PW。

每次展开预处理线程后，第一件事就是各线程检查相应的PW是否已创建。没创建就立刻创建。检查到已创建就不需要重新创建了。

在Preprocessor的worker_func_persistent中（这个方法似乎是关键的预处理逻辑），我们可以得到worker_id, label, data_ptr, data_size，这些就是我们每次执行预处理操作所需的新信息。

如果没记错的话，Preprocessor的持久线程池是在每个epoch的train阶段、val阶段各启动一次。由于不确定用户会否在第一次train之前就调用val，所以，train和val阶段启动线程池之后都必须检查PW是否已创建。

然后就是参数、内存。我们不需要在PW中使用内存池，因为我们实际上并不需要频繁申请释放内存。直接一次性申请一大块内存然后自己分块管理会比较简洁。

关于PW更新参数的方法，我比较建议是给PW类设计一个void update_parameters(const PreprocessWorkerParameter& param)。PreprocessWorkerParameter是一个结构体（这个结构体就放在跟PW同一个头文件就好，很简单），我们需要让Preprocessor类存储一个这样的结构体，每个phase（train或val）的间歇，在多线程没有展开的时候，Preprocessor可以更新这个结构体，把最新的信息写进去。到了多线程展开后，在检查PW已创建之后，每个线程就调用PW的update_parameters方法，把最新的参数更新到PW对象内部，然后才开始这个phase的预处理任务。

PW在创建时，构造函数需要包含很多信息。比如，线程ID，这个是要一次性写入、不再修改的。然后就是workshop的情况。也就是D、A/B、T区的大小（字节数）。如果大小是0就表示不分配这个区。注意这里AB区必须是一样大的，所以只传一个参数即可。然后就是要传入某个关于预处理操作列表的变量，我没想好，比如指向vector<PreprocessOperation>的指针？这里得说明一下，因为train和val的预处理是不同的，所以可能需要两个列表/队列/vector之类的来表示训练阶段和验证阶段的预处理操作。我们常打这样的比方：Preprocessor好比工厂，PW好比工人，PW里面分配的内存（我们称之为workshop）相当于工作台，而train和val的两套PreprocessOperation就相当于工人的两套工具。

然后，它还需要在构造函数中，一次性地、从GlobalRegistry把一些重要的基本信息保存到本地，包括：local_batch_size、global_batch_size、sdmp_factor、using_cpvs、num_train_samples、num_val_samples、world_size、max_resolution。

---







