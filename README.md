# Paddle_Mmdet2pytorch

## 为了获得输出特征为 x4,x8,x8,x8 的 resent_vd （而不是 x4,x8,x16,x32 的 resnet），就像 deeplab v3 等一些分割模型用的 backbone 一样（使用了空洞卷积）。
不知道 pytorch 官方提供的 resnet 是否可以直接使用，所以从 paddle 和 mmdet 中找，因为他们都提供了 resnet_vd 的代码和预训练模型。

目前，对于 paddle 的只有转化后的代码，没有把 paddle 提供的预训练模型转过来。对于 mmdet 既有转化代码，也有转化的 pretrained。

test_mmdet2pytorch.py : mmdet -> pytorch
test_paddle2pytorch.py : paddle -> pytorch

---
## 参考博客
1. Pytorch查看，输出，打印模型各个网络层的名字和参数
<https://blog.csdn.net/u014261408/article/details/120996966> 
2. PaddlePaddle与Pytoch模型参数之间相互赋值的实现方法 https://zhuanlan.zhihu.com/p/188744602 
3. https://cloud.tencent.com/developer/ask/sof/807229