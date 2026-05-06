# 语义先验增强的无监督多聚焦图像融合模型 Design 文档

## 0. 文档说明

本文档基于当前阶段 README、模型代码设计思路、训练方式、损失函数设计、数据集规划以及后续优化方向整理而成。该文档不是简单的现状记录，而是面向后续模型优化、论文方法章节撰写、实验设计、实施开发和阶段性复盘的设计依据。

当前模型已经完成大部分主体结构，包括 CNN-Swin 混合编码、语义先验引导、窗口注意力偏置、mask 决策图生成和无监督损失训练。本文档在此基础上给出完整设计，同时保留必要的结构升级预案。

------

# 1. 设计目标

## 1.1 任务目标

本项目面向多聚焦图像融合任务，即输入同一场景下两张具有不同聚焦区域的图像，生成一张同时保留两张源图像清晰区域的全聚焦融合图像。

当前任务形式为：

```text
Input:
  img1: 远焦图像或近焦图像之一
  img2: 与 img1 互补的另一张远/近焦图像

Output:
  fused_image: 融合图像
  mask: 融合决策图 / 权重图
  mask_logits: 未归一化决策图
```

模型最终通过如下公式进行融合：

```text
fused_image = mask * img1 + (1 - mask) * img2
```

其中，mask 接近 1 表示该位置更多采用 img1，mask 接近 0 表示该位置更多采用 img2。

## 1.2 方法定位

当前模型定位为：

> 语义先验增强的无监督多聚焦图像融合方法

该定位包含两层含义：

1. 模型训练不依赖真实全聚焦标签图像，因此属于无监督融合框架；
2. 模型引入 DeepLabv3 语义置信度与低级聚焦差异共同构成先验图 M，并将 M 注入 Swin Transformer 的窗口注意力 logits 中，因此不是纯无先验的无监督方法，而是先验增强的无监督融合方法。

## 1.3 核心设计目标

当前模型设计需要同时满足以下目标：

1. 保留源图像中的清晰纹理和边缘细节；
2. 保持融合图像的整体结构一致性和视觉自然性；
3. 利用 CNN 提取局部高频细节；
4. 利用 Swin Transformer 建模局部窗口和跨窗口上下文；
5. 利用语义先验图 M 引导注意力关注远近焦差异明显区域；
6. 通过 mask 生成可解释的像素级融合决策；
7. 在无监督损失约束下稳定训练；
8. 为后续细胞显微图像融合适配保留扩展空间。

------

# 2. 总体架构设计

## 2.1 当前模型名称

英文建议名称：

> Semantic-Prior-Guided CNN-Swin Hybrid Fusion Network

中文建议名称：

> 语义先验引导的 CNN-Swin 混合多聚焦图像融合网络

如果后续将真实跨图像注意力作为核心贡献，可进一步命名为：

> Semantic-Prior-Guided Cross-Attention Fusion Network

## 2.2 整体模块组成

当前模型由以下核心模块构成：

1. 双输入远近焦图像模块；
2. 语义/聚焦差异先验生成模块 SemanticPriorM；
3. Stem 初始特征映射模块；
4. 四阶段 Dense CNN 局部特征编码模块；
5. 四阶段 Prior-Guided Swin Token Mixer；
6. Semantic-Biased Window Self-Attention；
7. Spatial Enhancement Module；
8. Bottleneck Cross-Feature Attention Fusion；
9. U-Net 风格多尺度 mask 解码模块；
10. Mask-Guided Image Reconstruction；
11. ImprovedFusionLoss 无监督损失模块。

## 2.3 总体数据流

```text
img1 / img2
   │
   ├── SemanticPriorM(img1, img2) → M_full
   │
   ├── Stem(img1), Stem(img2)
   │
   ├── Stage 1: Dense CNN + tr1(M1) + Spatial Enhance
   │
   ├── Downsample
   │
   ├── Stage 2: Dense CNN + tr2(M2) + Spatial Enhance
   │
   ├── Downsample
   │
   ├── Stage 3: Dense CNN + tr3(M3) + Spatial Enhance
   │
   ├── Downsample
   │
   ├── Stage 4: Dense CNN + tr4(M4) + Spatial Enhance
   │
   ├── Bottleneck Cross-Attention Fusion
   │
   ├── Decoder: up3/dec3 → up2/dec2 → up1/dec1
   │
   ├── mask_head → mask_logits → sigmoid → mask
   │
   └── fused_image = mask * img1 + (1 - mask) * img2
```

------

# 3. 核心模块设计

## 3.1 双输入远近焦图像模块

### 3.1.1 输入定义

```text
img1: (B, 3, H, W)
img2: (B, 3, H, W)
```

通常 H = W = 512，像素归一化到 [0, 1]。

### 3.1.2 设计动机

多聚焦图像融合不是单图像增强任务，而是双图像互补信息选择任务。模型必须同时比较 img1 和 img2 中同一位置或邻域的清晰度差异。

### 3.1.3 设计作用

该模块提供双路输入，后续网络分别提取局部特征，并通过拼接、注意力和 mask 决策机制建模二者互补关系。

------

## 3.2 SemanticPriorM：语义/聚焦差异先验模块

### 3.2.1 模块定位

SemanticPriorM 是当前模型中体现“语义先验增强”的关键模块。其输出单通道先验图 M，用于指导 Swin Transformer 窗口注意力。

### 3.2.2 当前先验构造逻辑

当前先验图 M 综合两类信息：

1. 低级聚焦差异：LoG / Laplacian / Gradient 差异；
2. 语义边界信息：DeepLabv3 语义置信度图及其边缘响应。

简化表达为：

```text
D = |LoG(img1) - LoG(img2)| 或 |Grad(img1) - Grad(img2)|
S_conf = DeepLabv3(img1) 的语义置信图
S_edge = |∇S_conf|
M = Normalize(D * (1 + λ * S_edge)) ^ γ
```

### 3.2.3 设计动机

仅依靠 CNN 或梯度损失容易让模型过度关注局部高频差异，例如噪声、纹理或合成模糊边界；而仅依靠 Transformer 又可能缺乏细节敏感性。因此引入 M 的目标是让网络优先关注：

- 两张图像清晰度差异明显的位置；
- 前景/背景边界；
- 语义结构边缘；
- 聚焦/散焦切换区域。

### 3.2.4 与主干网络的关系

M_full 在全分辨率下生成，然后下采样到四个 stage：

```text
M_full → M1 → tr1
       → M2 → tr2
       → M3 → tr3
       → M4 → tr4
```

其中 tr1~tr4 是多尺度 Swin Token Mixer。M 不直接参与最终融合公式，而是在注意力 logits 中改变 token 之间的注意力分布。

### 3.2.5 当前风险

当前 DeepLabv3 先验默认来自 img1，可能导致先验偏向 img1。对于远近焦图像，img1 的模糊区域会影响语义置信图，因此后续建议验证：

```text
DeepLabv3(img1)
DeepLabv3((img1 + img2) / 2)
max(DeepLabv3(img1), DeepLabv3(img2))
```

------

## 3.3 Dense CNN 局部特征编码模块

### 3.3.1 当前结构

模型包含四个 CNN stage：

```text
cnn_stage1
cnn_stage2
cnn_stage3
cnn_stage4
```

每个 stage 由 DenseConvBlock 组成，通过多层卷积与特征拼接实现局部特征复用。

### 3.3.2 设计动机

多聚焦图像的聚焦区域通常表现为更强的高频响应，包括：

- 边缘；
- 纹理；
- 细胞边界；
- 物体轮廓；
- 局部梯度变化。

CNN 具备局部感受野和参数共享特性，适合提取这些局部高频信息。

### 3.3.3 作用

CNN 分支分别对 img1 和 img2 的特征进行局部编码：

```text
f1 = cnn_stage(f1)
f2 = cnn_stage(f2)
```

CNN 输出与 Transformer 输出共同生成 guide feature，进一步增强两路特征。

### 3.3.4 后续优化空间

1. 针对细胞图像加入更适合显微图像的局部结构模块；
2. 尝试使用深度可分离卷积降低复杂度；
3. 引入边缘增强卷积或 Laplacian/Sobel 引导分支；
4. 使用更强的多尺度 Dense Block 或 Res-Dense Block。

------

## 3.4 Multi-Scale Prior-Guided Swin Token Mixer

### 3.4.1 当前结构

模型包含四个 Swin Token Mixer：

```text
tr1, tr2, tr3, tr4
```

每个模块包含：

1. PatchEmbed2D；
2. SwinTransformerBlock2D，shift_size = 0；
3. SwinTransformerBlock2D，shift_size = window_size / 2；
4. LayerNorm；
5. 1×1 Conv 回投影。

### 3.4.2 设计动机

CNN 擅长局部纹理，但难以建模较大范围内的区域一致性。多聚焦图像融合中，一个完整前景或背景区域通常应保持一致选择。Swin Transformer 能够通过窗口注意力和移位窗口注意力建模局部窗口和跨窗口关系。

### 3.4.3 数据流

每个 stage 中：

```text
cat = concat(f1, f2)
tr = PatchTokenMixer(cat, M_stage)
tr = tr_proj(tr)
tr = interpolate(tr, size=f1.shape)
guide = SpatialEnhance((f1 + f2 + tr) / 3)
f1 = f1 + guide
f2 = f2 + guide
```

### 3.4.4 模块作用

该模块负责：

1. 建模 img1/img2 拼接特征的上下文关系；
2. 利用 M_stage 引导 Swin 注意力关注关键区域；
3. 提供区域级一致性约束；
4. 缓解 mask 碎片化问题。

------

## 3.5 Semantic-Biased Window Self-Attention

### 3.5.1 当前注意力形式

标准 Swin 窗口注意力为：

```text
Attention = Softmax(QK^T + RelativePositionBias) V
```

当前模型修改为：

```text
Attention = Softmax(QK^T + RelativePositionBias + β · SemanticBias(M)) V
```

其中：

```text
SemanticBias_ij = 0.5 * (m_i + m_j)
```

### 3.5.2 设计动机

将 M 加入 softmax 之前的 logits，而不是简单乘在 feature 上，可以直接影响注意力竞争过程。M 高的 token 在注意力分配中获得更高权重。

### 3.5.3 当前优点

1. 机制简单；
2. 易于解释；
3. 方便做消融实验；
4. 不改变主干结构；
5. 可以通过 prior_beta 控制先验强度。

### 3.5.4 当前问题

当前 bias 形式较温和，可能不足以显著改变注意力分布。后续可考虑：

```text
log-bias: log(0.5 * (m_i + m_j) + eps)
product-bias: m_i * m_j
difference-bias: -|m_i - m_j|
gated-bias: learnable_gate * SemanticBias(M)
```

------

## 3.6 Spatial Enhancement Module

### 3.6.1 当前结构

SpatialEnhanceModule 通过卷积生成空间权重：

```text
w = sigmoid(proj(x))
out = x * w + x
```

### 3.6.2 设计动机

CNN 和 Transformer 输出融合后，不同空间位置的重要性不同。空间增强模块进一步重标定 guide feature，使其更关注聚焦差异区域。

### 3.6.3 作用

该模块生成增强后的 guide，并残差注入两路特征：

```text
guide = EM((f1 + f2 + tr) / 3)
f1 = f1 + guide
f2 = f2 + guide
```

### 3.6.4 后续优化空间

1. 将普通空间注意力升级为通道+空间联合注意力；
2. 引入轻量 CBAM / ECA / BAM；
3. 将 M 也作为输入参与 EM；
4. 对细胞图像单独设计边缘敏感空间增强模块。

------

## 3.7 Bottleneck Cross-Feature Attention Fusion

### 3.7.1 当前实现

当前 stage4 后：

```text
cat4 = concat(s4_f1, s4_f2)
b = bottleneck_embed(cat4)
t = flatten(b)
t1 = t
t2 = flip(t)
z = cross_fuse(t1, t2)
z = bottleneck_back(z)
```

### 3.7.2 设计动机

该模块希望在低分辨率高语义层面对远焦/近焦特征进行跨域交互，增强互补信息建模能力。

### 3.7.3 当前不足

当前 `t2 = torch.flip(t)` 只是构造非对称 token 的简化方式，并不是真正从 img1 和 img2 分别构建 token。因此，该模块目前更像高层特征扰动注意力，而不是真正的双图像 cross-attention。

### 3.7.4 设计升级方案

建议后续改为真实双分支 token：

```text
t1 = proj1(s4_f1).flatten(2).transpose(1, 2)
t2 = proj2(s4_f2).flatten(2).transpose(1, 2)
z = CrossAttentionFuse(t1, t2)
```

该改动适合作为后续结构升级重点，也适合作为论文方法贡献。

------

## 3.8 多尺度 mask 解码模块

### 3.8.1 当前结构

当前 decoder 流程已确认：

```text
z
→ up3 + s3_g → dec3
→ up2 + s2_g → dec2
→ up1 + s1_g → dec1
→ mask_head
→ mask_logits
→ sigmoid
→ mask
```

### 3.8.2 设计动机

融合决策需要像素级空间精度，仅依靠高层特征无法恢复细粒度边界。多尺度解码模块利用 s1_g、s2_g、s3_g 中的浅中层信息恢复 mask 空间细节。

### 3.8.3 当前问题

已观察到当前 mask 可视化仍存在问题，可能包括：

- 边界不够贴合聚焦区域；
- 局部过于平滑；
- 某些纹理区域碎片化；
- 部分样本偏向某一张源图像；
- 聚焦/散焦边界处决策不稳定。

### 3.8.4 优化方向

1. 加入 edge-aware mask loss；
2. 引入 prior-mask alignment loss；
3. 优化 mask TV 与 bin loss 权重；
4. 增加 full-resolution refinement block；
5. 加入边界增强分支。

------

## 3.9 Mask-Guided Image Reconstruction

### 3.9.1 当前融合公式

```text
fused_image = mask * img1 + (1 - mask) * img2
```

### 3.9.2 设计动机

该设计将图像融合约束为源图像的加权选择，避免模型凭空生成内容，提高融合结果可解释性。

### 3.9.3 风险

该方式高度依赖 mask 质量。如果 mask 错误，则融合图像会直接出现错误区域选择、边界模糊或局部伪影。

------

# 4. 损失函数设计

## 4.1 当前损失函数

当前主要使用 ImprovedFusionLoss：

```text
L = λ_detail · L_detail
  + λ_ssim · L_ssim
  + λ_bin · L_mask_bin
  + λ_tv · L_mask_tv
```

其中：

```text
L_detail: LoG 细节保持损失
L_ssim: 基于 soft target 的 SSIM 结构损失
L_mask_bin: mask 二值化正则
L_mask_tv: mask TV 平滑正则
```

## 4.2 LoG 细节损失

```text
R1 = |LoG(img1)|
R2 = |LoG(img2)|
Rf = |LoG(fused)|
Rmax = max(R1, R2)
L_detail = ||Rf - Rmax||_1
```

作用：鼓励融合图像继承源图像中更强的边缘和纹理响应。

## 4.3 Soft Target SSIM 损失

```text
w = |LoG(img1)| / (|LoG(img1)| + |LoG(img2)| + eps)
target_soft = w * img1 + (1 - w) * img2
L_ssim = 1 - SSIM(fused, target_soft)
```

作用：在细节保持之外约束整体结构一致性。

## 4.4 Mask Bin Loss

```text
L_mask_bin = mean(mask * (1 - mask))
```

作用：鼓励 mask 接近 0/1，避免平均融合。

## 4.5 Mask TV Loss

```text
L_mask_tv = mean(|mask_h_diff|) + mean(|mask_w_diff|)
```

作用：抑制 mask 毛刺和碎片化。

## 4.6 当前损失权重

当前使用过的配置为：

```text
w_detail = 2.0
w_ssim = 0.5
w_mask_bin = 0.05
w_mask_tv = 0.02
gaussian_ksize = 5
gaussian_sigma = 1.0
ssim_ksize = 11
ssim_sigma = 1.5
```

【待确认】上述权重是否作为最终固定配置。

## 4.7 损失函数优化预案

由于后续将大量加入细胞图像，当前损失函数可能需要调整。可考虑：

1. 细胞边界增强损失；
2. 局部对比度损失；
3. 多尺度 LoG / Sobel 混合损失；
4. mask-prior alignment loss；
5. 细胞结构一致性损失；
6. 针对荧光图像的强度保持损失。

推荐新增 prior-mask 对齐损失：

```text
L_prior_align = ||∇mask - ∇M||_1
```

形成如下闭环：

```text
SemanticPriorM → Attention Bias → Feature Enhancement → Mask → Fusion
                    ↑                                  ↓
               Prior-Mask Alignment Loss
```

------

# 5. 数据集与训练策略设计

## 5.1 当前数据规模

当前训练集约 100 对远近焦图像，主要为常见物品远近焦图像。

该阶段数据主要用于：

1. 验证模型结构；
2. 检查 forward 和 loss 是否稳定；
3. 初步观察 mask 是否有效；
4. 验证语义先验是否带来可观察变化。

## 5.2 当前数据限制

100 对图像不足以支撑模型泛化能力结论，特别是当前模型包含 DeepLabv3、Swin Transformer 和多尺度 CNN，参数表达能力较强，存在小数据过拟合风险。

## 5.3 后续数据计划

后续计划大量加入真实细胞远近焦图像。细胞数据加入后，模型设计重点将发生变化：

1. 自然图像语义先验可能不再可靠；
2. DeepLabv3 对细胞语义可能失效；
3. 细胞边界、细胞核、细胞质纹理将成为重点；
4. 损失函数需要更重视细胞边缘和局部结构。

## 5.4 合成数据定位

当前合成远近焦数据不作为最终实验结论主要依据，主要用于前期结构调试。后续真实细胞远近焦照片将作为更重要的数据来源。

------

# 6. 当前问题诊断

## 6.1 结构层问题

1. Bottleneck cross-attention 不是真正的双图像 cross-attention；
2. semantic bias 形式较温和，先验影响可能不足；
3. DeepLabv3 先验可能偏向 img1；
4. 当前 decoder 对 mask 边界恢复能力仍需增强；
5. CNN 与 Transformer 的融合主要通过 guide 注入，交互深度仍可增强。

## 6.2 训练层问题

1. 训练数据规模偏小；
2. mask 可视化已经发现问题；
3. 当前 loss 没有直接约束 M 和 mask 的一致性；
4. TV loss 和 bin loss 权重需要围绕 mask 可视化调参；
5. LoG 细节损失可能对噪声或合成边界敏感。

## 6.3 数据层问题

1. 常见物品图像与细胞图像存在明显域差异；
2. DeepLabv3 对细胞图像适配性未知；
3. 合成模糊数据可能引入捷径学习；
4. 真实细胞远近焦数据尚未加入大规模训练。

## 6.4 论文表达层问题

如果模型定位为语义先验增强方法，则需要通过消融实验证明：

1. 语义先验确实有效；
2. M 注入注意力比普通特征乘权更有效；
3. prior_beta 的变化会影响融合结果；
4. M 与 mask 之间存在可解释关联。

------

# 7. 优化设计方案

## 7.1 短期优化设计

### 7.1.1 可视化与诊断工具

目标：判断模型问题来自先验、mask、loss 还是数据。

实现：

1. 保存 fused_image；
2. 保存 mask；
3. 保存 M_full；
4. 保存 M1~M4；
5. 统计 mask 均值、方差、二值化比例；
6. 计算 mask 与 M 的相关性。

预期效果：明确后续优化方向。

是否作为论文贡献：否，但可用于实验分析图。

### 7.1.2 prior_beta 消融

目标：验证 M 是否真正影响注意力。

实验设置：

```text
prior_beta = 0
prior_beta = 0.5
prior_beta = 1.0
prior_beta = 2.0
learnable_prior = True
```

预期效果：确定语义先验强度对 mask 和 fused image 的影响。

是否作为论文贡献：适合作为消融实验。

### 7.1.3 语义输入方式改进

目标：减少 img1 单图像语义输入带来的偏置。

已确认设计：

```text
S_conf = DeepLabv3((img1 + img2) / 2)
```

设计理由：

1. 多聚焦融合任务输入的是一对互补图像，单独使用 img1 可能使语义先验偏向某一张源图像；
2. `(img1 + img2) / 2` 能在一定程度上综合两张源图像的结构信息，使语义置信图更中立；
3. 该方案改动较小，便于快速实验验证。

后续可作为消融对比的候选方案：

```text
S_conf = DeepLabv3(img1)
S_conf = DeepLabv3(img2)
S_conf = DeepLabv3((img1 + img2) / 2)
S_conf = max(DeepLabv3(img1), DeepLabv3(img2))
```

预期效果：使语义先验更加中立，更适合远近焦互补输入。

是否作为论文贡献：可作为先验构造消融。

### 7.1.4 调整 mask loss 权重

目标：缓解 mask 过平滑、碎片化或偏向问题。

建议实验：

```text
w_mask_bin: 0.01 / 0.05 / 0.1
w_mask_tv: 0.0 / 0.01 / 0.02 / 0.05
```

预期效果：找到 mask 清晰度与连续性的平衡。

------

## 7.2 中期优化设计

### 7.2.1 真实双图像 Cross-Attention

目标：替换当前 `torch.flip(t)` 的简化方案，实现真正的 img1 ↔ img2 高层互补特征交互。

已确认设计：需要将 bottleneck cross-attention 改为真实双分支 cross-attention。

当前方案的问题：

```text
t1 = t
t2 = flip(t)
```

该方式虽然能制造非对称输入，但 t1 和 t2 本质上仍来自同一个拼接特征，不是真正来自 img1 和 img2 的独立 token 表达。

目标方案：

```text
t1 = Proj1(s4_f1).flatten(2).transpose(1, 2)
t2 = Proj2(s4_f2).flatten(2).transpose(1, 2)

z12 = Attention(Q=t1, K=t2, V=t2)
z21 = Attention(Q=t2, K=t1, V=t1)
z = Fuse(z12, z21)
```

推荐融合方式：

```text
z = z12 + z21
```

或：

```text
z = Conv/MLP(concat(z12, z21))
```

设计动机：

1. img1 与 img2 分别代表不同焦平面的清晰信息；
2. 真实 cross-attention 可以显式建模两张图像之间的互补关系；
3. 该模块可以使高层语义区域在两张图像之间进行信息对齐；
4. 对 mask 的区域级决策有潜在提升。

预期效果：增强远近焦互补信息建模，减少 mask 偏向单张源图像的问题。

论文价值：强，适合作为结构创新点。

### 7.2.2 改进 Semantic Bias 形式

目标：提高先验对注意力的有效影响。

候选：

```text
Sum Bias: 0.5 * (m_i + m_j)
Product Bias: m_i * m_j
Log Bias: log(0.5 * (m_i + m_j) + eps)
Difference Bias: -|m_i - m_j|
Hybrid Bias: α * Sum + β * Product
```

预期效果：让注意力更聚焦于语义/聚焦差异区域。

论文价值：中等，可作为注意力机制改进或消融。

### 7.2.3 加入 Prior-Mask Alignment Loss

目标：增强语义/结构先验 M 与最终决策图 mask 之间的闭环关系。

已确认设计：可以加入 Prior-Mask Alignment Loss。

推荐第一版实现：

```text
L_prior = ||∇mask - ∇M||_1
```

完整损失形式可扩展为：

```text
L = L_fusion + λ_prior · L_prior
```

其中：

```text
L_fusion = λ_detail · L_detail
         + λ_ssim · L_ssim
         + λ_bin · L_mask_bin
         + λ_tv · L_mask_tv
```

设计动机：

1. 当前 M 已经通过 attention logits 影响特征，但损失函数中没有显式约束 M 与 mask 的关系；
2. 如果论文主创新点是“语义先验引导注意力”，则需要证明先验不仅参与中间注意力，也能影响最终决策图；
3. 通过约束 mask 边界与 M 边界一致，可以增强模型可解释性。

实现注意事项：

1. `λ_prior` 初始值不宜过大，建议从 `0.01 ~ 0.05` 开始；
2. 如果 M 在细胞图像上不稳定，应先使用低级结构先验 M，而不是 DeepLabv3 语义先验；
3. 该项应作为可开关配置，方便消融实验。

预期效果：改善 mask 边界，提高先验和决策图之间的一致性。

论文价值：较强，适合作为训练策略贡献。

### 7.2.4 加强 Decoder 边界恢复能力

目标：解决 mask 边界不贴合问题。

设计：

1. 增加 full-resolution refinement block；
2. 在 decoder 中引入边缘特征；
3. 将 M_full 拼接到 mask_head 前；
4. 增加浅层 skip 连接。

预期效果：提升聚焦/散焦边界融合质量。

------

## 7.3 长期优化设计

### 7.3.1 细胞图像专用先验与 DeepLabv3 使用策略

目标：解决 DeepLabv3 自然图像语义先验在细胞图像上可能失效的问题。

已确认策略：细胞图像阶段优先测试低级结构先验，之后再加入 DeepLabv3 或其他语义模型。

第一阶段优先测试的低级结构先验包括：

```text
LoG difference prior
Gradient difference prior
Sobel edge prior
Local contrast prior
Multi-scale focus measure prior
```

推荐先验形式：

```text
M = Normalize(|LoG(img1) - LoG(img2)|)
```

或：

```text
M = Normalize(α · |LoG(img1) - LoG(img2)| + β · |Grad(img1) - Grad(img2)|)
```

第二阶段再测试语义/分割先验：

1. DeepLabv3；
2. CellPose 先验；
3. SAM / MedSAM 先验；
4. 细胞核分割模型；
5. 自训练显微图像边界检测模型。

设计理由：

1. 细胞图像与自然图像存在显著域差异；
2. DeepLabv3 对自然图像语义有效，但不一定能理解细胞结构；
3. 细胞多聚焦融合首先依赖边界、纹理和局部清晰度，而不一定依赖自然语义类别；
4. 低级结构先验更稳定，也更容易解释为聚焦差异先验。

论文价值：强。该方向可以将论文应用背景明确落到细胞图像，并形成“结构先验 → 语义先验 → 细胞应用”的渐进式实验逻辑。

### 7.3.2 细胞图像专用损失函数

目标：更好评价细胞边界、细胞核、细胞纹理。

候选：

1. Cell boundary loss；
2. Nuclear contrast loss；
3. Local contrast preservation loss；
4. Multi-scale gradient loss；
5. Fluorescence intensity consistency loss。

预期效果：提高细胞融合图像对后续分析任务的可用性。

### 7.3.3 下游任务驱动评估

目标：不只评价图像视觉质量，还评价融合图像对细胞分析任务的帮助。

候选任务：

1. 细胞分割；
2. 细胞计数；
3. 细胞核定位；
4. 荧光强度分析；
5. 形态学参数提取。

论文价值：强，能够证明方法实际应用价值。

------

# 8. 实验设计方案

## 8.1 基础对比实验

对比对象：

1. 原始 CNN-Swin 模型；
2. 无语义先验模型；
3. 当前 SemanticPriorM 模型；
4. 改进 Cross-Attention 模型；
5. 改进 loss 模型。

## 8.2 消融实验

| 实验项         | 设置                        | 目的               |
| -------------- | --------------------------- | ------------------ |
| Semantic Prior | on / off                    | 验证先验有效性     |
| prior_beta     | 0 / 0.5 / 1 / 2 / learnable | 验证先验强度       |
| Semantic Input | img1 / avg / max            | 验证语义偏置       |
| Bias Form      | sum / product / log / diff  | 验证注意力偏置形式 |
| Cross-Attn     | flip / true cross           | 验证跨图像交互     |
| Loss           | current / +prior / +edge    | 验证训练约束       |
| Data           | object / cell               | 验证跨域泛化       |

## 8.3 可视化实验

必须保存：

1. img1；
2. img2；
3. fused_image；
4. mask；
5. M_full；
6. M 与 mask 叠加图；
7. 局部放大图；
8. 聚焦边界对比图。

## 8.4 评价指标

当前可使用：

1. EN；
2. SSIM；
3. PSNR；
4. MI；
5. AG；
6. SF；
7. Qabf；
8. edge preservation。

细胞图像后续可补充：

1. 细胞边界清晰度；
2. 细胞分割 Dice；
3. 细胞计数准确率；
4. 细胞核定位误差；
5. 下游分析一致性。

------

# 9. 实施计划

## 9.1 阶段一：现有模型验证

1. 清理 forward 中重复 fused_image 语句；
2. 确认 up0/dec0 是否存在未使用参数；
3. 保存 M_full 和 mask 可视化；
4. 对 Lytro/MFFW 做初步可视化对比；
5. 记录 mask 问题类型。

## 9.2 阶段二：先验有效性实验

1. use_semantic_prior = False；
2. prior_beta = 0；
3. prior_beta 多组实验；
4. img1 / avg / max 三种 semantic input；
5. 分析 M 与 mask 关系。

## 9.3 阶段三：结构升级

1. 实现真实 s4_f1 ↔ s4_f2 cross-attention；
2. 测试新的 semantic bias；
3. 增加 prior-mask alignment loss；
4. 优化 decoder 边界恢复。

## 9.4 阶段四：细胞数据适配

1. 收集真实细胞远近焦数据；
2. 测试 DeepLabv3 先验是否有效；
3. 设计细胞专用 prior；
4. 设计细胞图像损失；
5. 增加下游任务评估。

## 9.5 阶段五：论文整理

1. 整理方法图；
2. 整理消融实验；
3. 整理主观对比图；
4. 整理客观指标表；
5. 写方法章节；
6. 写实验章节；
7. 写局限性与展望。

------

# 10. 关键设计决策

## 10.1 已确认的设计决策

| 设计项                     | 当前决策                                                     |
| -------------------------- | ------------------------------------------------------------ |
| 训练方式                   | 无监督训练                                                   |
| 方法定位                   | 语义先验增强的无监督多聚焦图像融合方法                       |
| 论文主创新点               | 语义先验引导注意力机制                                       |
| 应用背景                   | 细胞图像多聚焦融合                                           |
| DeepLabv3                  | 当前作为语义先验候选模块，后续在低级结构先验验证后再加入测试 |
| ONNX                       | 当前不考虑                                                   |
| 融合方式                   | mask 加权融合                                                |
| 主干结构                   | CNN + Swin 多尺度混合编码                                    |
| Bottleneck Cross-Attention | 确认需要改为真实双分支 cross-attention                       |
| 语义输入方式               | 确认改为 `(img1 + img2) / 2`                                 |
| Prior-Mask Alignment Loss  | 确认可以加入，作为先验与 mask 决策图的闭环约束               |
| 细胞图像先验策略           | 优先测试低级结构先验，之后再加入 DeepLabv3 或其他语义模型    |

## 10.2 当前待验证的设计

| 设计项                            | 状态           |
| --------------------------------- | -------------- |
| DeepLabv3 对细胞图像是否有效      | 后续验证       |
| 低级结构先验在细胞图像中的有效性  | 优先验证       |
| M 与 mask 是否一致                | 待验证         |
| prior_beta 最优值                 | 待验证         |
| semantic bias 最优形式            | 待验证         |
| 当前 mask 问题根因                | 后续测试后分析 |
| 真实 cross-attention 是否提升效果 | 待实验         |
| 细胞图像专用 loss                 | 待设计         |

## 10.3 本轮决策对模型设计的影响

本轮决策后，模型优化路线从“泛化式结构探索”收敛为以下主线：

```text
低级结构先验 / 语义先验 M
        ↓
Prior-Guided Swin Attention
        ↓
真实双分支 Cross-Attention
        ↓
Prior-Mask Alignment
        ↓
Mask-Guided Fusion
        ↓
细胞图像多聚焦融合应用
```

这意味着后续模型优化不再优先追求简单堆叠模块，而是围绕“先验是否有效影响注意力、注意力是否改善 mask、mask 是否提升融合质量”建立完整闭环。

------

# 11. 风险分析

## 11.1 技术风险

1. 语义先验在细胞图像上失效；
2. 模型复杂度较高导致小数据过拟合；
3. mask 决策图质量不足；
4. 先验影响注意力但无法有效传导到 mask；
5. 当前 loss 对细胞图像不适配。

## 11.2 实验风险

1. 当前训练集规模小；
2. Lytro/MFFW 测试结果可能不能代表细胞图像效果；
3. 客观指标与主观质量可能不一致；
4. 消融实验过多，需要控制变量。

## 11.3 论文表达风险

1. 如果不做先验消融，语义先验贡献难以证明；
2. 如果不展示 M 与 mask，可解释性不足；
3. 如果 DeepLabv3 不能适配细胞图像，方法定位需要调整；
4. 如果 mask 可视化较差，决策图驱动融合的说服力会下降。

------

# 12. 当前建议优先级

## P0：必须完成

1. mask、M_full、fused_image 可视化；
2. prior on/off 消融；
3. prior_beta 消融；
4. 分析 mask 当前问题类型；
5. 整理 Lytro/MFFW 已测试结果。

## P1：强烈建议完成

1. semantic input 改为 avg(img1,img2)；
2. 实现真实 cross-attention；
3. 增加 prior-mask alignment loss；
4. 测试不同 semantic bias 形式。

## P2：细胞数据到位后完成

1. DeepLabv3 细胞图像适配性测试；
2. 细胞专用先验；
3. 细胞图像损失函数；
4. 下游任务评估。

------

# 13. 待讨论问题与当前决策结果

本节原本用于列出进入下一轮模型结构大改前需要确认的问题。当前已完成第一轮决策，结果如下。

## 13.1 是否优先将 bottleneck cross-attention 改为真实双分支 cross-attention？

决策：是。

后续将当前 `torch.flip(t)` 的简化实现替换为基于 `s4_f1` 和 `s4_f2` 的真实双分支 cross-attention。

设计方向：

```text
t1 = Proj1(s4_f1)
t2 = Proj2(s4_f2)
z12 = Attention(Q=t1, K=t2, V=t2)
z21 = Attention(Q=t2, K=t1, V=t1)
z = Fuse(z12, z21)
```

## 13.2 DeepLabv3 的语义输入是否立即改成 `(img1 + img2) / 2`？

决策：是。

该修改用于减少单独使用 img1 造成的语义偏置，使语义先验更适合远近焦互补输入。

## 13.3 是否需要马上加入 `L_prior_align = ||∇mask - ∇M||_1`？

决策：可以加入。

该损失作为先验 M 与最终 mask 决策图之间的对齐约束，有助于形成“先验 → 注意力 → mask → 融合图像”的闭环。

建议初始权重：

```text
λ_prior = 0.01 ~ 0.05
```

## 13.4 当前 mask 的主要问题是什么？

决策：暂时不处理，后续测试后再分析。

后续需要通过 mask 可视化判断其主要问题类型：

1. 过平滑；
2. 碎片化；
3. 偏向 img1；
4. 偏向 img2；
5. 边界错位；
6. 大面积灰度化。

## 13.5 细胞图像到位后，是否继续保留 DeepLabv3？

决策：先优先测试低级结构先验，之后再加入 DeepLabv3。

原因：细胞图像与自然图像差异较大，DeepLabv3 的自然图像语义先验不一定适配细胞图像。低级结构先验如 LoG、Gradient、Sobel 和 local contrast 更符合细胞图像聚焦差异建模。

## 13.6 论文主创新点如何定位？

决策：论文主创新点定位为“语义先验引导注意力”，应用背景定位为“细胞图像多聚焦融合”。

推荐论文表达：

> 本文提出一种语义先验增强的无监督多聚焦图像融合方法，通过将聚焦差异先验与语义/结构先验注入 Swin Transformer 的窗口注意力机制，引导模型在无真实全聚焦标签的条件下生成更可靠的融合决策图。该方法面向细胞图像多焦平面融合场景，旨在提升显微图像中细胞边界、纹理和结构信息的保留能力。

------

# 14. 结论

当前模型已经具备较完整的语义先验增强无监督多聚焦融合框架，包括 CNN-Swin 多尺度混合编码、语义先验引导窗口注意力、mask 驱动融合和无监督损失约束。后续优化的关键不在于简单堆叠模块，而在于围绕以下三条主线形成闭环：

```text
先验 M 是否有效 → 注意力是否被引导 → mask 是否更可靠 → fused 是否更清晰
```

当前最重要的优化方向包括：

1. 验证语义先验有效性；
2. 修正 bottleneck cross-attention 的跨图像建模方式；
3. 提升 mask 决策图质量；
4. 为真实细胞远近焦图像适配先验和损失函数；
5. 通过消融实验支撑论文创新点。

该 Design 文档后续可继续扩展为：

- 模型实施文档；
- 论文方法章节；
- 实验设计文档；
- 代码重构计划；
- 阶段性汇报材料。