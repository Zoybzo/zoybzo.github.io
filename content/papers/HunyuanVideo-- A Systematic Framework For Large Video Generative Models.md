---
share: "true"
data: 2025-03-07
tags:
  - 视频生成
  - CV
  - 🤔
title: HunyuanVideo-- A Systematic Framework For Large Video Generative Models
slug: 22:33
series: 视频生成
cover.image: 
dir: papers
---
# Paper
## Overview
- HunyuanVideo 是一款开源的视频生成基座模型，打破了目前的开源模型远远弱于闭源模型的情形。
- Hunyuan 在论文中提出了多种模型，包括文生视频、视频生音频、图生视频，以及姿态控制、音频控制、表情控制等多种下游微调任务。
- 完整的 pipeline 如下所示： 
{{< figure src="/images/Pasted image 20250307161744.png"  width="656" height="250">}}
- 在 Pipeline 中主要包含三个部分：数据处理、基座模型训练以及下游任务。
## Detail
### `T2V` 模型
#### `3D VAE` 
- 与以往的视频生成模型不同，Hunyuan 选择自己从头训练一个 `3D VAE` 模型。从实验上来看，Hunyuan 的 `3D VAE` 有更高的 PSNR，也就是有更好的重构效果。
- 模型的 PSNR 实验效果如下：
![image-2.png](image-2.png)
- `3DVAE` 的模型结构如下：
![image.png](image.png)
##### 训练
- 训练数据采用 `视频：图片=4:1` 比例。
- 使用多种损失共同构成损失函数：L1损失+KL损失+LPIPS损失+GAN损失。全都是和图片重构相关的损失函数。
![image-1.png](image-1.png)
- 训练策略上，采取了渐进式训练策略，从低分辨率短视频逐渐过渡到高分辨率长视频。
- 为了提升模型对于高速视频的重建能力，会从 `[1,8]` 中随机采样一个数字作为采样间隔，使用该采样间隔从视频帧中进行均匀采样，采样结果用于训练。
	- 这样的帧率应该就不一样了呀？ #🤔
	- 通过 Mask 来实现输入帧率不变但实际帧率变化？
##### 推理
- 高分辨率长视频的推理往往难以在单张GPU上完成，为了解决这个问题，论文提出了**时空瓦片策略**（spatial-temporal tiling strategy）（很形象的名字，视频首尾相连，类似于房顶的瓦片），将视频拆成有部分重叠的片段，对每个片段单独编解码，并使用线性组合处理重叠部分。
- 直接在推理中使用瓦片策略会导致训练和推理中出现不一致的情况（重叠部分的处理导致的），为了解决这个问题，论文在训练之后引入了一个微调阶段，**在微调阶段中随机决定是否采用瓦片策略**，保证训练和推理的一致性。
- 按照 T 维度（temporal）进行操作的代码如下：
```python
def temporal_tiled_encode(self, x: torch.FloatTensor,  
                          return_dict: bool = True) -> AutoencoderKLOutput:  
  
    B, C, T, H, W = x.shape  
    # 减去重叠部分后的长度，用作步长  
    overlap_size = int(  
        self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))  
    # 前后帧的重叠长度  
    blend_extent = int(  
        self.tile_latent_min_tsize * self.tile_overlap_factor)  
    # 最大长度  
    t_limit = self.tile_latent_min_tsize - blend_extent  
  
    # Split the video into tiles and encode them separately.  
    row = []  
    for i in range(0, T, overlap_size):  
        # 每一个瓦片的长度为 tile_sample_min_tsize        tile = x[:, :, i: i + self.tile_sample_min_tsize + 1, :, :]  
        # 处理空间维度  
        if self.use_spatial_tiling \  
                and (tile.shape[-1] > self.tile_sample_min_size \  
                     or tile.shape[-2] > self.tile_sample_min_size):  
            tile = self.spatial_tiled_encode(tile, return_moments=True)  
        else:
            tile = self.encoder(tile)  
            tile = self.quant_conv(tile)  
        if i > 0:  
            tile = tile[:, :, 1:, :, :]  
        row.append(tile)  
    result_row = []  
    for i, tile in enumerate(row):  
        if i > 0:  
            # 线性组合当前帧与前一帧的重叠部分  
            tile = self.blend_t(row[i - 1], tile, blend_extent)  
            result_row.append(tile[:, :, :t_limit, :, :])  
        else:  
            # 第一帧正常返回  
            result_row.append(tile[:, :, :t_limit + 1, :, :])  
  
    moments = torch.cat(result_row, dim=2)  
    posterior = DiagonalGaussianDistribution(moments)  
  
    if not return_dict:  
        return (posterior,)  
  
    return AutoencoderKLOutput(latent_dist=posterior)
```
#### Trm Block
- Trm 使用 unified Full Attention 机制，不再将 spatial 和 temporal 分开计算。（像 CogVideoX 一样）
- Trm 结构如下：
![image-3.png](image-3.png)
- Trm 部分的超参数如下：
![image-4.png](image-4.png)
##### 输入处理
- 对于视频，使用 `3DVAE` 转换成 Latents
- 对于文本，首先使用 LLM（代码中使用的是Llama3）编码成 Embedding，捕获**精细的语义信息**，同时使用 CLIP 提取池化的文本表示，包含**全局信息**。
##### 主要结构
- 主要结构采用了从 Dual-Stream DiT 到 Single-Stream DiT 的设计。（类似于 Flux）
- 代码中实际上用的就是 `Flux.1` 中的 Double 和 Single 网络
##### 位置编码
- 将 RoPE 扩展到了三维，具体来说是先分别计算 T、H、W 三个维度的旋转频率矩阵，之后将 C 划分成三部分 (dt, dh, dw)，将每一段乘以对应的频率，并将结果 concat，得到最终的 RoPE
	- 如何划分的呢？
#### Text Encoder
- 使用 MLLM 代替 CLIP/T5 作为文本编码器。
	- 相比于 T5，MLLM 在特征空间有更好的图文对齐；
	- 相比于 CLIP，MLLM 在图片细节理解和复杂推理上更强；
	- MLLM可以遵循 prompt 去编码文本，将注意力更多地放在关键信息上。
- MLLM 基于因果注意，而 T5-XXL 则利用双向注意，**双向注意可以为扩散模型提供更好的文本指导**，因此使用 Refineer 增强 MLLM 输出的文本特征。
![image-5.png](image-5.png)
- CLIP 提供的全局信息同样非常重要。类似于Flux和SDv3，通过 scale/shift/gate 的方式将 CLIP 的全局信息添加到模型中。
#### Scaling Law

#### 模型预训练
#### 模型加速
#### 模型性能评估
### `V2A` 模型
### `I2V` 模型
### 人像动画生成
