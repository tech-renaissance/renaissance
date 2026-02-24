### （三）FastRandomResizedCrop

RandomResizedCrop通常选取面积范围为0.08~1.00。这个分布不是均匀分布，而是对数分布，也就是先对两个边界值求自然对数，在其范围内取均匀分布随机数，再然后取自然指数。这样的结果就是，更倾向于Crop出更小的图形。

在我们的SDMP/局部解码方案下，RandomResizedCrop的行为是：如果sdmp_factor=1（即不使用SDMP），那么解码的时候就是完全的局部解码，只解码要Crop的那部分；如果sdmp_factor>1，那么就解码整张图片，然后对其进行若干次Crop。两种优化没法用在一起。

那接下来是我们的FastRandomResizedCrop。它是局部解码与SDMP的有机结合。如果sdmp_factor=1，那么FastRandomResizedCrop的行为与RandomResizedCrop一致，就是只解码需要Crop的那个局部。特别之处就在于sdmp_factor＞1的情况。首先，当用户输入面积范围[a,b]时，计算边界值的三次方根c=sqrt3(a)、d=sqrt3(b)。对于sdmp_factor=2，你先在[c,d]之间取均匀分布随机数p，然后取p²，这就是要局部解码的面积比例。在解码出来的区域里面，再在[c,d]之间取均匀分布随机数q，以q为比例，对解码出来的区域进行两次Crop，分别完成后面的预处理流程，保存到缓冲区或传送到输出。对于sdmp_factor≥4，你先在[c,d]之间取均匀分布随机数p，这就是要局部解码的面积比例（而不是p²）。在解码出来的区域里面，再在[c,d]之间取均匀分布随机数q，以q²为比例，对解码出来的区域进行多次Crop，分别完成后面的预处理流程，保存到缓冲区或传送到输出。而对于sdmp_factor=3的情形，我们设置一个宏，叫CROP_SCHEME_2，它为true的时候，sdmp_factor=3的情形采取sdmp_factor=2的策略，它为false的时候，sdmp_factor=3的情形采取sdmp_factor=4的策略。这个方案就是执行两步随机，先确定解码窗口，再确定随机Crop窗口。唯一需要注意的是长宽比，我认为长宽比应该只随机一次，以免增加Crop失败的概率。这个策略在sdmp_factor＞1的时候，平均速度必定比RandomResizedCrop快，并且在sdmp_factor = 2或3的时候优势最明显。在提高速度的同时，不会明显丧失随机性。我们必须指出，通过这种策略Crop出来的面积分布是三次方根分布（p²q或pq²），与对数分布虽然有微小的不同，但差距非常小，几乎可以忽略不计，三次方根分布也是倾向于Crop出较小的图形，但整体上比对数分布略大一点。如果我们使用了FastRandomResizedCrop来利用SDMP和局部解码优化，预计可以在第一个epoch解码时间上打七折，然后随后的1~2个epoch又能直接免除加载、解码、预处理，这样带来的加速幅度将是非常可观的。

SDMP优化，针对的是JPEG输入，并且需要足够大的内存。