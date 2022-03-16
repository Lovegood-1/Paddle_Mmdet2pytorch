# Paddle_Mmdet2pytorch

## 为了获得输出特征为 x4,x8,x8,x8 的 resent_vd （而不是 x4,x8,x16,x32 的 resnet），就像 deeplab v3 等一些分割模型用的 backbone 一样（使用了空洞卷积）。
不知道 pytorch 官方提供的 resnet 是否可以直接使用，所以从 paddle 和 mmdet 中找，因为他们都提供了 resnet_vd 的代码和预训练模型。

目前，对于 paddle 的只有转化后的代码，没有把 paddle 提供的预训练模型转过来。对于 mmdet 既有转化代码，也有转化的 pretrained。