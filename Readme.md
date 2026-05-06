~~~markdown
# 模型结构阶段性总结

## 1. 项目背景与研究目标

本项目面向 **多聚焦图像融合（Multi-Focus Image Fusion, MFIF）** 任务。该任务的目标是将同一场景下具有不同聚焦区域的两张图像进行融合，生成一张尽可能同时保留远焦区域和近焦区域清晰信息的全聚焦图像。

当前模型主要处理一对远近焦图像：

- 输入图像 A：近焦图像或远焦图像之一；
- 输入图像 B：与 A 互补的另一张远近焦图像；
- 输出结果：
  - 融合图像 `fused_image`；
  - 决策图 / 权重图 `mask`；
  - 未归一化决策图 `mask_logits`。

当前模型采用 **无监督训练方式**。由于多聚焦图像融合任务通常缺少真实的全聚焦标签图像，因此模型不依赖人工标注的融合结果，而是通过损失函数约束融合图像在梯度、结构、纹理、亮度和 mask 形态等方面尽可能继承两张源图像中的清晰信息。

当前阶段模型的核心思想可以概括为：

> 利用 CNN 提取局部高频细节，利用 Swin Transformer 建模全局上下文关系，再通过语义先验图 M 引导注意力机制关注更可能发生聚焦差异的区域，最终通过 mask 以可解释方式完成远近焦图像融合。

---

## 2. 当前模型整体架构

当前模型可以命名为：

> **Semantic-Prior-Guided CNN-Swin Hybrid Fusion Network**

中文可表述为：

> **语义先验引导的 CNN-Swin 混合多聚焦图像融合网络**

整体架构由以下部分组成：

1. 输入远近焦图像对；
2. 语义先验引导模块 `SemanticPriorM`；
3. Stem 初始特征映射模块；
4. 四阶段 CNN 局部特征提取模块；
5. 四阶段 Swin Transformer Token Mixer 模块；
6. 空间增强模块 `SpatialEnhanceModule`；
7. Bottleneck Cross-Attention 融合模块；
8. U-Net 风格多尺度解码模块；
9. mask 生成模块；
10. mask 驱动图像融合模块。

整体流程如下：

```text
Near-Focus Image / Far-Focus Image
        │
        ├── SemanticPriorM → Prior Map M
        │
        ├── Stem Feature Projection
        │
        ├── Stage 1: Dense CNN + Prior-guided Swin Token Mixer + Spatial Enhance
        │
        ├── Stage 2: Downsample + Dense CNN + Prior-guided Swin Token Mixer + Spatial Enhance
        │
        ├── Stage 3: Downsample + Dense CNN + Prior-guided Swin Token Mixer + Spatial Enhance
        │
        ├── Stage 4: Downsample + Dense CNN + Prior-guided Swin Token Mixer + Spatial Enhance
        │
        ├── Bottleneck Cross-Attention Fusion
        │
        ├── Multi-scale Decoder with Skip Features
        │
        ├── Mask Head → mask_logits → sigmoid → mask
        │
        └── fused = mask * img1 + (1 - mask) * img2
~~~

当前模型不是直接回归融合图像，而是先生成一张融合决策图 `mask`，再通过如下方式得到融合图像：

```python
fused_image = mask * img1 + (1 - mask) * img2
```

这种方式具有较强的可解释性：
当 `mask` 接近 1 时，该位置更多选择图像 A；当 `mask` 接近 0 时，该位置更多选择图像 B。

------

## 3. 模型核心模块设计

### 3.1 输入图像处理模块

#### 模块名称

**Dual-Input Multi-Focus Image Pair Module**

#### 输入

```text
img1: (B, 3, H, W)
img2: (B, 3, H, W)
```

其中 `img1` 和 `img2` 分别表示一对远近焦图像。图像通常被归一化到 `[0, 1]`。

#### 设计动机

多聚焦图像融合的本质是从两张源图像中选择清晰区域并组合，因此模型必须同时感知两张图像之间的差异与互补关系。单独处理某一张图像无法判断该区域是否比另一张图像更清晰。

#### 模块作用

该模块为后续所有模块提供基础输入。模型后续会分别提取两张图像的局部纹理特征，并在 Transformer 分支中通过拼接特征建模二者之间的上下文关系。

------

### 3.2 语义先验引导模块：SemanticPriorM

#### 模块名称

**Semantic-Guided Focus Difference Prior Module**

#### 当前实现

当前模型中定义了 `SemanticPriorM`，用于生成一张单通道先验图 `M`。该模块结合低级视觉差异和语义分割置信信息，构造一个用于引导 Transformer 注意力的 prior map。

其核心形式为：

```text
D = |LoG(img1) - LoG(img2)| 或 |Grad(img1) - Grad(img2)|
S_conf = DeepLabv3(img1) 的语义置信度
S_edge = |∇S_conf|
M = Normalize( D * (1 + λ * S_edge) ) ^ γ
```

#### 设计动机

仅依赖 CNN 或梯度损失时，模型容易只关注局部高频差异，例如纹理、噪声或人工模糊边界；仅依赖 Transformer 时，又可能出现局部细节恢复不足的问题。因此，引入先验图 `M` 的目标是让模型在注意力建模阶段更加关注：

- 两张源图像差异明显的区域；
- 语义结构边界区域；
- 可能发生聚焦/散焦切换的区域；
- 前景与背景交界处等高价值决策区域。

#### 模块作用

`SemanticPriorM` 不直接参与最终图像融合，而是作为注意力偏置注入到 Swin Transformer 的窗口注意力中。它本质上提供的是一种 **区域重要性先验**。

#### 与其他模块的配合方式

`M_full` 在输入分辨率下生成，然后被下采样到不同 stage 的尺度：

```text
M_full → M1 → tr1
       → M2 → tr2
       → M3 → tr3
       → M4 → tr4
```

每个尺度的 Swin Token Mixer 都会接收对应尺度的 `M_stage`。

#### 当前注意事项

当前代码中 `S_conf` 默认由 `img1` 输入 DeepLabv3 得到。该设计实现简单，但可能引入对 `img1` 的偏置。后续可以考虑改为：

```python
S_conf = DeepLabv3((img1 + img2) / 2)
```

或：

```python
S_conf = max(DeepLabv3(img1), DeepLabv3(img2))
```

以减弱单图像语义偏置。

------

### 3.3 CNN 局部特征提取模块

#### 模块名称

**Dense Local Detail Encoding Module**

#### 当前实现

模型使用四个 CNN stage：

```python
cnn_stage1
cnn_stage2
cnn_stage3
cnn_stage4
```

每个 stage 由两个 `DenseConvBlock` 组成。`DenseConvBlock` 内部通过多层卷积和特征拼接实现局部特征复用。

#### 设计动机

多聚焦图像融合高度依赖边缘、纹理、梯度等局部高频信息。CNN 具有天然的局部感受野优势，适合提取：

- 清晰边缘；
- 纹理细节；
- 局部梯度变化；
- 聚焦区域的高频响应。

#### 解决的问题

传统 Transformer 更擅长全局建模，但对细粒度边缘和局部纹理的建模能力相对不足。CNN 分支补充了这一不足，使模型能够更稳定地判断局部区域清晰度。

#### 模块作用

在每个尺度上，CNN 分别对 `img1` 与 `img2` 的特征进行局部编码：

```text
f1 = cnn_stage(f1)
f2 = cnn_stage(f2)
```

CNN 输出随后会与 Transformer 输出共同生成 guide feature，用于增强两路特征。

------

### 3.4 Swin Transformer 全局上下文建模模块

#### 模块名称

**Multi-Scale Prior-Guided Swin Token Mixer**

#### 当前实现

模型中包含四个 Transformer token mixer：

```python
tr1
tr2
tr3
tr4
```

它们均由 `PatchTokenMixer` 实现。每个 `PatchTokenMixer` 包含：

1. `PatchEmbed2D`；
2. 两个 `SwinTransformerBlock2D`；
3. `LayerNorm`；
4. `1×1 Conv` 特征回投影。

`PatchEmbed2D` 通过 stride 为 patch size 的卷积将特征图转换为 token 序列；Swin block 内部包含窗口注意力和移位窗口注意力。

#### 设计动机

CNN 主要关注局部纹理，而多聚焦图像融合不仅需要判断局部清晰度，还需要保证区域级一致性。例如一个前景物体通常应该整体来自同一张更清晰的源图像，而不是在局部产生碎片化选择。

Swin Transformer 的作用是：

- 建模局部窗口内 token 之间的关系；
- 通过 shift window 实现跨窗口信息交互；
- 捕获更大范围内的结构一致性；
- 结合先验图 M 强化关键区域的注意力响应。

#### 模块作用

在每个 stage 中，模型先将两路 CNN 特征拼接：

```python
cat = torch.cat([f1, f2], dim=1)
```

然后送入当前尺度的 `PatchTokenMixer`：

```python
tr = tr_mixer(cat, M=M_stage)
```

Transformer 输出经过 `tr_proj` 降维，再上采样回 CNN 特征尺寸：

```python
tr = tr_proj(tr)
tr = interpolate(tr, size=f1.shape[-2:])
```

随后与 CNN 特征共同生成增强 guide：

```python
guide = (f1 + f2 + tr) / 3
guide = SpatialEnhanceModule(guide)
```

最终 guide 以残差方式注入两路特征：

```python
f1 = f1 + guide
f2 = f2 + guide
```

#### 对最终融合效果的影响

该模块有助于缓解纯 CNN 模型容易出现的以下问题：

- mask 碎片化；
- 聚焦区域不连续；
- 大范围前景/背景判断不一致；
- 仅根据局部纹理产生错误选择。

------

### 3.5 语义先验对窗口注意力的影响机制

#### 模块名称

**Semantic-Biased Window Self-Attention**

#### 当前实现

当前模型在 `WindowAttention` 中将语义先验 `M_win` 注入到注意力 logits 中。

标准窗口注意力为：

```text
Attention = Softmax(QK^T + RelativePositionBias) V
```

当前模型改为：

```text
Attention = Softmax(QK^T + RelativePositionBias + β · SemanticBias(M)) V
```

其中：

```python
bias_ij = 0.5 * (m_i + m_j)
```

即窗口内两个 token 的 prior 值越高，其对应的注意力 logit 会越高。

#### 设计动机

如果只在特征层简单乘以先验图，先验可能只是局部增强特征幅值；而将先验加入 softmax 之前的 attention logits，相当于改变注意力竞争机制，使重要区域在注意力分配阶段获得更高权重。

#### 模块作用

该机制可以让模型在 Swin 注意力中更关注：

- 远近焦差异明显的区域；
- 语义边界区域；
- 前景/背景切换区域；
- 聚焦决策不确定区域。

#### 当前实现的优点

当前设计的优点是简单、稳定、容易解释：

```text
M 高 → attention logits 增强 → softmax 后注意力权重提高
```

#### 当前实现的潜在不足

当前 `bias_ij = 0.5 * (m_i + m_j)` 相对温和，可能导致先验影响不足。后续可以考虑：

```python
bias = torch.log(bias + eps)
```

或引入温度系数、pairwise difference bias，使先验对注意力的影响更加明确。

------

### 3.6 空间增强模块

#### 模块名称

**Spatial Enhancement Module**

#### 当前实现

`SpatialEnhanceModule` 通过卷积生成空间权重：

```python
w = sigmoid(proj(x))
out = x * w + x
```

#### 设计动机

Transformer 输出和 CNN 输出融合后，不同空间位置的重要性并不相同。空间增强模块通过学习空间注意力权重，使模型更关注关键区域。

#### 模块作用

在每个 stage 中，`SpatialEnhanceModule` 对 guide feature 进行增强，然后将其残差注入两路特征：

```text
guide = EM((f1 + f2 + tr) / 3)
f1 = f1 + guide
f2 = f2 + guide
```

#### 影响

该模块有助于增强：

- 聚焦区域；
- 纹理区域；
- 语义边界；
- 图像结构变化明显区域。

------

### 3.7 Bottleneck Cross-Attention 融合模块

#### 模块名称

**Bottleneck Cross-Feature Attention Fusion Module**

#### 当前实现

当前模型在 stage4 后，将 `s4_f1` 与 `s4_f2` 拼接：

```python
cat4 = torch.cat([s4_f1, s4_f2], dim=1)
```

然后经过 `bottleneck_embed` 得到 token 序列，并进入 `CrossAttentionFuse`。

当前代码中：

```python
t1 = t
t2 = torch.flip(t, dims=[1])
```

然后进行双向 cross-attention：

```python
a12 = MHA(query=t1, key=t2, value=t2)
a21 = MHA(query=t2, key=t1, value=t1)
z = a12 + a21
```

#### 设计动机

该模块的目标是让远焦图像和近焦图像在高层语义空间中发生信息交互，增强两张图像之间的互补关系建模。

#### 当前不足

当前实现中的 `t2 = torch.flip(t, dims=[1])` 更像是一种占位式扰动，并不是真正意义上的 `img1 token ↔ img2 token` 跨图像注意力。因此该模块虽然形式上是 cross-attention，但跨域交互还不够严格。

#### 后续建议

更合理的实现方式是分别从 `s4_f1` 和 `s4_f2` 生成 token：

```python
t1 = proj1(s4_f1).flatten(2).transpose(1, 2)
t2 = proj2(s4_f2).flatten(2).transpose(1, 2)
z = cross_fuse(t1, t2)
```

该修改更适合作为论文中的“跨图像互补特征融合模块”。

------

### 3.8 多尺度解码与决策图生成模块

#### 模块名称

**Multi-Scale Mask Decoding Module**

#### 当前设计

模型定义了：

```python
up3, up2, up1, up0
dec3, dec2, dec1, dec0
mask_head
```

设计意图是采用 U-Net 风格解码方式，逐级上采样并融合浅层 guide feature：

```text
z
→ up3 + s3_g
→ up2 + s2_g
→ up1 + s1_g
→ full-resolution refinement
→ mask_head
→ mask_logits
→ sigmoid
→ mask
```

#### 设计动机

多聚焦融合需要像素级或近像素级决策，因此仅有 bottleneck 高层特征是不够的。多尺度解码能够结合：

- 深层语义结构；
- 中层区域关系；
- 浅层边缘细节；
- 原图分辨率下的局部纹理。

当前模型在 bottleneck cross-attention 后，采用 U-Net 风格的多尺度解码结构恢复空间分辨率。具体而言，模型依次执行：

z → up3 + s3_g → dec3  
→ up2 + s2_g → dec2  
→ up1 + s1_g → dec1  
→ mask_head → mask_logits → sigmoid → mask

最终通过如下公式得到融合图像：

fused_image = mask * img1 + (1 - mask) * img2

其中，mask 表示融合决策图。当 mask 接近 1 时，模型倾向于选择 img1 的像素；当 mask 接近 0 时，模型倾向于选择 img2 的像素。该结构使模型输出具有较强可解释性。

(todo):当前 bottleneck cross-attention 用于增强高层融合特征，但其双域 token 构造方式仍较简化，后续可进一步改为基于 `s4_f1` 与 `s4_f2` 的真实跨图像注意力。

------

### 3.9 mask 驱动图像融合输出模块

#### 模块名称

**Mask-Guided Image Reconstruction Module**

#### 当前设计

模型最终通过 mask 对源图像进行线性组合：

```python
fused_image = mask * img1 + (1 - mask) * img2
```

#### 设计动机

相比直接让网络生成融合图像，mask 融合方式具有更强的物理可解释性：

- 输出内容来自源图像；
- 减少模型凭空生成纹理的风险；
- mask 可以可视化分析；
- 适合无监督训练；
- 便于论文中解释“决策图”。

#### 可能影响

该结构有助于提高融合结果的稳定性，但也对 mask 质量提出更高要求。如果 mask 边界过于平滑，会导致聚焦/散焦边界模糊；如果 mask 过于碎片化，会导致融合图像出现噪声或局部伪影。

------

## 4. 前向传播流程

当前模型的前向传播可总结如下：

```python
def forward(img1, img2):

    # 1. 计算语义先验图
    if use_semantic_prior:
        M_full = SemanticPriorM(img1, img2)
    else:
        M_full = None

    # 2. 初始特征映射
    f1 = stem(img1)
    f2 = stem(img2)

    # 3. Stage 1
    M1 = M_full
    s1_f1, s1_f2, s1_g = Stage1(f1, f2, M1)

    # 4. Stage 2
    d1_f1 = down(s1_f1)
    d1_f2 = down(s1_f2)
    M2 = resize(M_full, size=d1_f1)
    s2_f1, s2_f2, s2_g = Stage2(d1_f1, d1_f2, M2)

    # 5. Stage 3
    d2_f1 = down(s2_f1)
    d2_f2 = down(s2_f2)
    M3 = resize(M_full, size=d2_f1)
    s3_f1, s3_f2, s3_g = Stage3(d2_f1, d2_f2, M3)

    # 6. Stage 4
    d3_f1 = down(s3_f1)
    d3_f2 = down(s3_f2)
    M4 = resize(M_full, size=d3_f1)
    s4_f1, s4_f2, s4_g = Stage4(d3_f1, d3_f2, M4)

    # 7. Bottleneck attention fusion
    cat4 = concat(s4_f1, s4_f2)
    z = CrossAttention(cat4)

    # 8. Multi-scale decoder
    x = Decoder(z, s3_g, s2_g, s1_g)

    # 9. Generate mask
    mask_logits = mask_head(x)
    mask = sigmoid(mask_logits)

    # 10. Image fusion
    fused = mask * img1 + (1 - mask) * img2

    return fused, mask, mask_logits
```

其中每个 stage 的内部逻辑为：

```python
def Stage(f1, f2, M_stage):
    f1 = CNNStage(f1)
    f2 = CNNStage(f2)

    cat = concat(f1, f2)
    tr = PriorGuidedSwinTokenMixer(cat, M_stage)

    tr = project(tr)
    tr = upsample_to_CNN_size(tr)

    guide = SpatialEnhance((f1 + f2 + tr) / 3)

    f1 = f1 + guide
    f2 = f2 + guide

    return f1, f2, guide
```

------

```
## 5. 损失函数设计

当前模型采用无监督训练策略。由于多聚焦图像融合任务缺少真实全聚焦标签，因此损失函数不直接约束模型输出接近某一张 ground truth，而是从源图像中构造可优化目标，引导融合图像同时保留两张输入图像中的清晰区域。

当前主要使用的损失函数为：

```text
L = λ_detail · L_detail
  + λ_ssim · L_ssim
  + λ_bin · L_mask_bin
  + λ_tv · L_mask_tv
```

其中：

- `L_detail`：LoG 细节保持损失；
- `L_ssim`：基于 soft target 的结构相似性损失；
- `L_mask_bin`：mask 二值化正则；
- `L_mask_tv`：mask 空间平滑正则。

当前训练调用形式为：

```
loss, l_detail, l_ssim, l_bin, l_tv = criterion(img1, img2, fused, mask)
```

------

### 5.1 LoG 细节保持损失

#### 设计思想

多聚焦图像中，聚焦区域通常具有更强的边缘、纹理和高频响应，而散焦区域的高频细节会被模糊削弱。因此，当前损失函数使用 LoG（Laplacian of Gaussian）近似提取图像细节响应。

当前实现为：

```
LoG(x) = Laplacian(GaussianBlur(x))
```

具体步骤为：

1. 对输入图像进行固定高斯平滑；
2. 使用固定 3×3 拉普拉斯核计算二阶边缘响应；
3. 对 `img1`、`img2` 和 `fused` 分别计算 LoG 响应；
4. 使用两张源图像 LoG 幅值的逐像素最大值作为细节目标。

公式可表示为：

```
R1 = |LoG(img1)|
R2 = |LoG(img2)|
Rf = |LoG(fused)|

Rmax = max(R1, R2)

L_detail = || Rf - Rmax ||_1
```

#### 作用

该损失鼓励融合图像在边缘和纹理区域保留两张源图像中更清晰的一方，从而提升融合图像的锐度和细节表现。

#### 解决的问题

如果没有细节损失，模型可能倾向于生成结构上较平滑、视觉上较平均的融合结果，导致清晰区域边缘变弱或纹理丢失。

------

### 5.2 基于 soft target 的 SSIM 结构损失

#### 设计思想

仅使用 LoG 细节损失可能会过度强调局部高频，使融合图像在整体结构或亮度一致性上不稳定。因此当前损失函数进一步构造一个 soft target，用于指导融合图像保持合理的结构外观。

soft target 的构造方式为：

```
w = |LoG(img1)| / (|LoG(img1)| + |LoG(img2)| + eps)

target_soft = w · img1 + (1 - w) · img2
```

然后计算：

```
L_ssim = 1 - SSIM(fused, target_soft)
```

#### 作用

该损失使融合结果在整体结构上接近由清晰度权重构造出的软融合目标，避免模型只追求局部梯度而破坏图像整体观感。

#### 解决的问题

该损失主要用于缓解：

- 图像局部过锐；
- 结构不连续；
- 明暗不稳定；
- 融合结果与源图像整体外观不一致。

------

### 5.3 mask 二值化正则项

#### 设计思想

多聚焦图像融合本质上更接近“选择问题”：某个位置应该更多来自 img1，还是更多来自 img2。因此 mask 不宜在大量区域长期保持 0.5 附近，否则会变成简单平均融合，导致聚焦/散焦边界模糊。

当前二值化正则为：

```
L_mask_bin = mean(mask · (1 - mask))
```

当 `mask` 接近 0 或 1 时，该项较小；当 `mask` 接近 0.5 时，该项较大。

#### 作用

该损失鼓励 mask 更接近二值化决策，使模型在清晰区域选择上更加明确。

#### 解决的问题

该项主要用于减少：

- 大面积平均融合；
- mask 灰度化严重；
- 融合结果发糊；
- 远近焦边界不清晰。

------

### 5.4 mask TV 平滑正则项

#### 设计思想

如果只鼓励 mask 二值化，mask 可能出现大量孤立点或碎片化区域，导致融合图像中产生局部噪声和不自然跳变。因此当前损失加入 Total Variation 正则，约束 mask 在空间上保持连续。

当前 TV 损失为：

```
L_mask_tv =
mean(|mask[:, :, 1:, :] - mask[:, :, :-1, :]|)
+
mean(|mask[:, :, :, 1:] - mask[:, :, :, :-1]|)
```

#### 作用

该项鼓励 mask 在空间上更加平滑，减少毛刺、碎片和孤立噪声点。

#### 解决的问题

该项主要用于缓解：

- mask 过度碎片化；
- 融合图像局部闪烁或噪声；
- 小区域错误选择；
- 决策图不连续。

------

### 5.5 当前损失函数的整体作用

当前 `ImprovedFusionLoss` 的整体目标可以概括为：

```
LoG detail loss：保证清晰细节
SSIM soft target loss：保证结构一致性
mask bin loss：保证决策明确
mask TV loss：保证决策连续
```

该损失函数适合当前 mask-driven 无监督融合框架，因为它既约束最终融合图像，也约束中间决策图 mask 的形态。

------

### 5.6 当前训练中使用的损失权重

根据当前训练配置，损失权重曾设置为：

```
criterion = ImprovedFusionLoss(
    w_detail=2.0,
    w_ssim=0.5,
    w_mask_bin=0.05,
    w_mask_tv=0.02,
    gaussian_ksize=5,
    gaussian_sigma=1.0,
    ssim_ksize=11,
    ssim_sigma=1.5,
)
```

其中：

- `w_detail=2.0`：增强细节保持能力；
- `w_ssim=0.5`：保留结构一致性，但不过度压制锐度；
- `w_mask_bin=0.05`：轻度鼓励 mask 二值化；
- `w_mask_tv=0.02`：轻度平滑 mask，避免过度平滑导致边界模糊。

【待确认：以上是否为当前最终训练版本的固定权重。】

------

### 5.7 当前损失函数的潜在不足

尽管当前损失函数已经能够支持无监督训练，但仍存在以下优化空间：

#### 1. mask 与语义先验 M 尚未显式一致

当前语义先验 M 主要通过注意力 logits 影响特征建模，但损失函数中没有直接约束：

```
mask 是否与 M 在高差异区域保持一致
```

后续可以考虑增加：

```
L_prior = || edge(mask) - edge(M) ||
```

或：

```
L_prior = BCE(mask_uncertainty_region, M_high_region)
```

用于强化先验对最终决策图的解释力。

#### 2. LoG 细节损失可能对噪声敏感

LoG 能很好提取边缘和纹理，但也可能放大噪声或人工模糊边界。后续可以考虑：

- Sobel + LoG 混合；
- 多尺度 LoG；
- 引入局部对比度损失；
- 对 LoG 响应加置信权重。

#### 3. TV loss 可能导致 mask 边界过平滑

TV 正则有利于减少碎片，但如果权重过大，会使聚焦/散焦边界变得过度平滑，影响边界区域融合质量。

后续需要通过可视化 mask 判断：

```
mask 是否过于平滑？
mask 边界是否对齐聚焦区域边界？
mask 是否存在大面积 0.5 区域？
```

#### 4. 当前 soft target 由 LoG 权重构造，仍属于低级视觉先验

当前 soft target 主要依赖 LoG 幅值，没有显式引入语义区域一致性。因此对于低纹理但语义重要的区域，可能约束不足。

后续可以考虑引入：

- semantic region consistency loss；
- foreground-aware mask regularization；
- edge-aware smoothness loss。



你这个 loss 目前是**合理的第一版无监督融合损失**，它的优点是：

```
清晰度目标明确
实现稳定
完全可微
适合 mask-driven 融合框架
容易做消融实验
```

但如果你后面要把“语义先验引导”作为论文重点，那么下一步最好补一个与语义先验相关的损失或分析，否则读者可能会问：

> 你既然引入了 SemanticPriorM，为什么损失函数里没有任何项直接约束 mask 和 prior 的关系？

所以后续可以考虑一个轻量增强版：

```
L = 原 ImprovedFusionLoss + λ_prior · L_prior_align
```

其中 `L_prior_align` 可以先做得很简单，例如：

```
L_prior_align = || ∇mask - ∇M ||_1
```

这会让论文逻辑更闭环：

```
语义先验 M 生成
→ M 影响注意力
→ M 辅助 mask 边界
→ mask 引导融合图像
```

（todo）：之后损失函数的选择会有一部分更改，更适合于评价细胞图像的损失函数



------

## 6. 数据集与训练策略

### 6.1 当前训练数据规模

当前阶段训练集规模约为 **100 对远近焦图像**。数据主要来源于常见物品场景，每组样本包含两张具有互补清晰区域的远近焦图像。

当前数据规模仍处于模型验证和结构调试阶段，适合用于：

- 验证模型 forward 流程是否正确；
- 观察 mask 是否能够形成有效决策；
- 初步比较不同损失函数和结构模块的影响；
- 检查语义先验模块是否能够参与注意力建模。

但从深度学习模型训练角度看，100 对图像仍然偏少，暂时不足以充分支撑模型泛化能力评估。

### 6.2 当前数据类型

当前数据主要为常见物品远近焦图像，包括自然场景、日常物体或普通视觉场景中的远近焦图像对。

这类数据的优势是：

- 语义结构较清晰；
- 前景/背景关系明显；
- 适合验证语义先验模块；
- 适合观察 mask 是否能区分远近焦区域。

其不足是：

- 与最终显微/细胞图像应用场景仍存在域差异；
- 普通物品图像的纹理、边缘和语义结构与细胞图像不同；
- 基于 DeepLabv3 的语义先验更适合自然图像，对细胞图像的语义解释能力可能不足。

### 6.3 后续数据扩充计划

后续计划继续增加训练数据，重点包括：

1. 扩充常见物品远近焦图像对；
2. 大量加入细胞显微图像；
3. 构造真实或合成的细胞多焦平面图像对；
4. 逐步建立以细胞图像为主的训练集、验证集和测试集。

对于细胞图像，应重点关注：

- 细胞边界；
- 细胞核区域；
- 细胞质纹理；
- 荧光/明场图像差异；
- 显微图像背景噪声；
- Z-stack 不同焦平面的清晰区域变化。

### 6.4 数据域差异问题

当前模型引入了 DeepLabv3 语义先验，该先验通常基于自然图像语义训练。当训练数据逐步转向细胞显微图像时，需要重点评估语义先验模块是否仍然有效。

可能存在的问题包括：

- DeepLabv3 无法正确理解细胞图像语义；
- 语义置信度图在细胞图像上可能退化；
- 语义边缘不一定对应细胞结构边界；
- 先验 M 可能更多依赖 LoG/Gradient 差异，而不是语义分割信息。

因此，在加入大量细胞图像后，需要单独进行以下实验：

| 实验                         | 目的                                         |
| ---------------------------- | -------------------------------------------- |
| 不使用语义先验               | 验证 CNN+Swin 主干在细胞图像上的基础能力     |
| 使用当前 DeepLabv3 先验      | 验证自然图像语义先验是否迁移有效             |
| 仅使用 LoG/Gradient 差异先验 | 判断低级视觉先验是否比语义先验更适合细胞图像 |
| 使用细胞分割模型作为先验     | 验证领域专用先验是否更有效                   |

### 6.5 当前训练策略

当前训练仍处于小规模验证阶段。建议当前阶段训练目标不要过早追求指标最优，而应优先关注：

- loss 是否稳定下降；
- mask 是否能形成清晰区域选择；
- fused image 是否出现明显伪影；
- prior map M 是否与聚焦差异区域相关；
- 加入语义先验前后是否存在可观察变化。

当前训练数据规模较小时，不建议直接得出模型泛化能力结论。

（todo）：当前语义先验模块在自然图像上具备较好解释性，但其在细胞显微图像上的有效性仍需进一步验证。对于细胞图像，后续可能需要替换为细胞分割模型、显微图像结构先验，或仅使用低级聚焦差异先验。

------

## 7. 当前模型的阶段性创新点

### 7.1 已实现创新点：CNN 与 Swin Transformer 的多尺度混合建模

当前模型在四个尺度上同时使用 CNN 和 Swin Transformer。CNN 负责局部高频细节，Swin Transformer 负责窗口内和跨窗口的上下文建模。

该设计有助于同时保留：

- 局部清晰边缘；
- 纹理细节；
- 区域级结构一致性；
- 前景/背景大范围关系。

### 7.2 已实现创新点：语义先验引导的窗口注意力机制

当前模型将语义先验图 `M` 作为 logits-level bias 注入窗口注意力中：

```text
QK^T + RelativePositionBias + β · SemanticBias(M)
```

该设计使先验不是简单作为后处理权重，而是在 softmax 之前直接影响注意力竞争过程。

### 7.3 已实现创新点：决策图驱动的可解释图像融合

模型输出 mask，并通过 mask 对两张源图像进行线性组合。这种方式具备较强可解释性，适合展示融合决策区域，也方便进行可视化分析。

### 7.4 潜在创新点：低级聚焦差异与高级语义先验的联合建模

`SemanticPriorM` 同时使用 Laplacian/Gradient 差异和 DeepLabv3 语义置信度边缘。这一设计具备一定创新潜力，但仍需通过消融实验验证其有效性。

### 7.5 潜在创新点：多尺度 prior-guided token mixing

当前 `M_full` 被插值到 `tr1~tr4` 四个尺度，并分别影响各尺度 Swin 注意力。这可以作为“多尺度先验引导注意力”的潜在创新点。

### 7.6 后续可强化方向：真实跨图像 Cross-Attention 融合

当前 bottleneck cross-attention 使用 `torch.flip` 构造 t2，严格意义上还不是完整的双图像 cross-attention。后续如果改为 `s4_f1 ↔ s4_f2` 的真实 cross-attention，可以作为更强的结构贡献。

------

## 8. 当前存在的问题

### 8.1 当前不足

#### 8.1.1 decoder 末尾代码需要确认

当前上传版本中，forward 的 decoder 末尾疑似不完整，`mask` 与 `mask_logits` 在可见代码中没有被定义就被使用。

【待补充：确认最终训练代码是否包含完整 mask_head 逻辑。】

#### 8.1.2 语义先验可能偏向 img1

当前语义置信度由 `img1` 输入 DeepLabv3 得到。对于远近焦图像对，单图像语义可能受到该图像模糊区域影响，并导致 prior map 偏向 img1。

#### 8.1.3 semantic bias 形式相对温和

当前 `bias_ij = 0.5 * (m_i + m_j)`，虽然稳定，但可能不足以显著改变注意力分布。

#### 8.1.4 bottleneck cross-attention 不够严格

当前 `t2 = torch.flip(t, dims=[1])` 不能严格表示 `img1` 与 `img2` 的双向跨域注意力。

#### 8.1.5 mask 边界可能过平滑

TV loss 有助于平滑 mask，但过强时可能导致边界不够锐利，影响聚焦/散焦边界处的融合质量。

#### 8.1.6 数据构造方式会影响泛化

如果合成模糊区域过于规则，模型可能学习合成数据的几何模式，而不是学习真实聚焦差异。

#### 8.1.7 模型复杂度较高

DeepLabv3 + 多尺度 Swin + CNN encoder 的结构较重，后续如果需要 ONNX 或实时部署，需要考虑轻量化。

------

## 9. 后续优化方向

### 9.1 短期优化

#### 9.1.1 修复或确认 decoder 末尾逻辑

**优化目标：**
确保模型 forward 完整且可稳定训练/推理。

**实现方式：**

```python
x = self.up1(x)
x = self.dec1(torch.cat([x, s1_g], dim=1))

x = self.dec0(torch.cat([x, stem_ref], dim=1))

mask_logits = self.mask_head(x)
mask = torch.sigmoid(mask_logits)
fused_image = mask * img1 + (1 - mask) * img2
```

**预期效果：**
保证模型结构与输出一致，避免训练和推理阶段出现未定义变量或尺寸错误。

**是否适合作为论文创新点：**
否，属于工程修复。

------

#### 9.1.2 将语义输入从 img1 改为双图像融合输入

**优化目标：**
减少语义先验对单张源图像的偏置。

**实现方式：**

```python
S_conf = self._semantic_conf((img1 + img2) / 2)
```

或：

```python
S_conf1 = self._semantic_conf(img1)
S_conf2 = self._semantic_conf(img2)
S_conf = torch.max(S_conf1, S_conf2)
```

**预期效果：**
使语义先验更加中立，更适合远近焦互补输入。

**是否适合作为论文创新点：**
可以作为小型改进点或消融实验点。

------

#### 9.1.3 调整 prior_beta

**优化目标：**
验证语义先验是否真正影响注意力。

**实现方式：**

- 设置 `prior_beta = 0` 作为无先验 baseline；
- 设置 `prior_beta = 0.5, 1.0, 2.0, 4.0` 做消融；
- 尝试 `learnable_prior=True`。

**预期效果：**
评估先验注入强度对融合质量和 mask 质量的影响。

**是否适合作为论文创新点：**
适合作为消融实验。

------

#### 9.1.4 可视化 prior map 与 mask

**优化目标：**
增强模型可解释性。

**实现方式：**

- 保存 `M_full`；
- 保存 `M1~M4`；
- 保存最终 `mask`；
- 对比 `M` 与 `mask` 的空间相关性。

**预期效果：**
判断先验是否真正引导了决策图生成。

**是否适合作为论文创新点：**
适合作为论文可视化分析。

------

### 9.2 中期优化

#### 9.2.1 将 bottleneck cross-attention 改为真实双分支 cross-attention

**优化目标：**
增强远焦/近焦图像之间的互补信息交互。

**实现方式：**

```python
t1 = proj1(s4_f1).flatten(2).transpose(1, 2)
t2 = proj2(s4_f2).flatten(2).transpose(1, 2)
z = cross_fuse(t1, t2)
```

**预期效果：**
更准确地建模两张源图像之间的互补关系。

**是否适合作为论文创新点：**
是，适合作为“跨图像互补注意力融合模块”。

------

#### 9.2.2 改进 semantic bias 形式

**优化目标：**
增强先验对注意力的实际影响。

**实现方式：**

当前：

```python
bias_ij = 0.5 * (m_i + m_j)
```

可尝试：

```python
bias = log(0.5 * (m_i + m_j) + eps)
```

或：

```python
bias_ij = -abs(m_i - m_j)
```

或：

```python
bias_ij = m_i * m_j
```

**预期效果：**
使注意力更聚焦于高置信区域或相似显著性区域。

**是否适合作为论文创新点：**
是，适合作为“先验引导注意力机制”的细化贡献。

------

#### 9.2.3 增加 mask 边界约束

**优化目标：**
改善聚焦/散焦边界处的融合质量。

**实现方式：**

加入边界相关损失：

```text
L_edge = || ∇mask - Normalize(|∇img1 - ∇img2|) ||
```

或约束 mask 边界与 prior map 边界一致。

**预期效果：**
减少边界模糊、重影和 halo。

**是否适合作为论文创新点：**
可以作为训练策略创新点。

------

#### 9.2.4 增加结构一致性约束

**优化目标：**
避免融合图像在结构上偏离源图像。

**实现方式：**

- MS-SSIM loss；
- gradient consistency loss；
- local contrast loss；
- frequency-domain detail loss。

**预期效果：**
增强融合图像整体视觉稳定性。

**是否适合作为论文创新点：**
适合作为损失函数改进点。

------

### 9.3 长期优化

#### 9.3.1 构建完整的语义先验引导融合框架

**优化目标：**
将语义先验从辅助模块提升为核心贡献。

**实现方式：**

- 使用双图像语义置信图；
- 使用 prior-guided cross-attention；
- 使用 mask-prior consistency loss；
- 使用多尺度 prior supervision。

**预期效果：**
形成完整的“语义先验—注意力—决策图”闭环。

**是否适合作为论文创新点：**
是，适合作为核心创新点。

------

#### 9.3.2 构建更真实的多聚焦数据生成机制

**优化目标：**
提升模型对真实多聚焦图像的泛化能力。

**实现方式：**

- 使用不规则软边界模糊区域；
- 控制 A/B 模糊区域尽量不重叠；
- 引入 defocus blur 圆盘核；
- 模拟散焦扩散效应；
- 使用多种 blur sigma；
- 进行数据构造消融实验。

**预期效果：**
减少模型对合成数据分布的过拟合。

**是否适合作为论文创新点：**
是，特别适合作为数据建模部分贡献。

------

#### 9.3.3 模型轻量化与 ONNX 部署

**优化目标：**
提升推理效率，为工程应用做准备。

**实现方式：**

- 替换 DeepLabv3 为轻量语义模型；
- 减少 Swin stage 数量；
- 减小 embed_dim；
- 使用可导出的 ONNX-friendly attention；
- 将部分操作替换为部署友好算子。

**预期效果：**
降低参数量和推理延迟。

**是否适合作为论文创新点：**
可作为工程应用价值补充，不一定作为核心算法创新点。

------

## 10. 后续论文写作可展开的内容

### 10.1 方法章节可展开内容

1. 问题定义：无监督多聚焦图像融合；
2. 整体网络框架；
3. CNN 局部细节编码模块；
4. Swin Transformer 全局上下文建模模块；
5. 语义先验图构造；
6. 语义先验注入窗口注意力机制；
7. Cross-Attention 融合模块；
8. mask 决策图生成；
9. 无监督损失函数设计；
10. 训练数据构造方法。

### 10.2 实验章节可展开内容

1. 与传统 MFIF 方法对比；
2. 与深度学习 MFIF 方法对比；
3. 在 Lytro / MFFW / 自建数据集上测试；
4. 主观视觉对比；
5. 客观指标对比；
6. mask 可视化；
7. prior map 可视化；
8. 消融实验。

### 10.3 推荐消融实验

| 消融项            | 对比设置                                         | 目的               |
| ----------------- | ------------------------------------------------ | ------------------ |
| 无 semantic prior | `use_semantic_prior=False`                       | 验证先验有效性     |
| prior_beta        | 0 / 0.5 / 1 / 2 / learnable                      | 验证先验强度       |
| semantic input    | img1 / avg(img1,img2) / max(conf1,conf2)         | 验证语义偏置问题   |
| bias 形式         | sum / product / log / difference                 | 验证注意力注入方式 |
| 去掉 Swin         | CNN-only                                         | 验证全局建模       |
| 去掉 CNN          | Transformer-only                                 | 验证局部细节       |
| 去掉 cross-attn   | no bottleneck fusion                             | 验证跨图像交互     |
| 去掉 mask TV      | no TV loss                                       | 验证平滑约束       |
| 不同数据生成方式  | rectangle / irregular / soft edge / defocus blur | 验证数据构造影响   |

------

## 11. 下一阶段工作计划

### 第一阶段：工程正确性确认

1. 确认 `forward` 末尾是否完整；
2. 确认 `mask_logits`、`mask` 的生成逻辑；
3. 确认输出尺寸与输入尺寸一致；
4. 跑通单张推理；
5. 可视化 `fused_image`、`mask`、`M_full`。

### 第二阶段：先验模块有效性验证

1. 对比 `use_semantic_prior=True/False`；
2. 对比 `prior_beta` 不同取值；
3. 保存注意力图或间接分析 attention response；
4. 可视化 prior map 与 mask 的空间关系。

### 第三阶段：结构优化

1. 将 bottleneck cross-attention 改为真实 `s4_f1 ↔ s4_f2`；
2. 改进 semantic bias 形式；
3. 加入 mask 边界损失；
4. 优化 decoder 细节恢复能力。

### 第四阶段：数据构造优化

1. 使用不规则单区域模糊；
2. 控制 A/B 模糊区域尽量不重叠；
3. 增加 defocus blur；
4. 设计数据构造消融实验。

### 第五阶段：论文实验整理

1. 指标计算；
2. 主观图对比；
3. mask 可视化；
4. prior map 可视化；
5. 消融实验表格；
6. 模型复杂度分析；
7. 下游任务或应用场景补充。

------

## 后续需要进一步确认的问题

1. 【待补充】合成远近焦数据中 blur kernel、sigma、区域面积范围的最终设置。

2. 【待补充】当前模型参数量、FLOPs、单张推理耗时。

3. 【已明确】是否已经在真实 Lytro / MFFW 图像上测试。

   1. 当前模型已经在真实多聚焦图像上进行过初步测试，包括 Lytro / MFFW 类型数据。测试结果可用于初步观察模型在非合成远近焦图像上的泛化表现。

      【待补充】具体测试样本数量、主观融合效果、客观评价指标与可视化结果。

4. 【已明确】当前 `mask` 可视化效果是否存在过平滑、碎片化或偏向某一张图像的问题。

   1. 当前模型的 mask 可视化结果仍存在一定问题，可能表现为：

      \- mask 边界不够贴合真实聚焦区域；
      \- 局部区域过于平滑；
      \- 某些纹理区域出现碎片化选择；
      \- 部分样本中 mask 可能偏向某一张源图像；
      \- 聚焦/散焦边界处存在不稳定决策。

      这说明当前模型虽然已经具备 mask 驱动融合能力，但决策图质量仍是后续优化重点。

5. 【已明确】是否计划将 DeepLabv3 先验模型保留在最终推理阶段，还是仅用于训练阶段辅助。

   1. 当前阶段计划在推理阶段保留 DeepLabv3 语义先验模块，使模型在训练和推理时均能利用语义先验图 M 参与注意力引导。

      但需要注意，DeepLabv3 通常基于自然图像语义训练，在细胞显微图像上的适配性仍需验证。如果后续实验发现语义先验在细胞图像上不稳定，后续可考虑两种替代方案：

      1. 将 DeepLabv3 仅作为训练阶段辅助模块，用于引导中间特征或 mask；
      2. 使用细胞图像专用分割模型或结构先验模型替代 DeepLabv3。

6. 【已明确】后续是否需要导出 ONNX；如果需要，需要单独评估 Swin window attention 的 ONNX 兼容性。

   1. 当前阶段暂不考虑 ONNX 导出与部署优化，后续工作重点仍放在模型结构、损失函数、数据集构造和实验验证上。

7. 【已明确】论文中是否将该模型定位为纯无监督方法、弱先验引导方法，还是语义先验增强的无监督融合方法。

   1. 当前模型定位为：

      **语义先验增强的无监督多聚焦图像融合方法**

      该定位强调两点：

      1. 模型训练不依赖真实全聚焦标签图像，因此仍属于无监督融合框架；
      2. 模型通过 DeepLabv3 和低级视觉差异构造语义/聚焦先验图 M，并将其注入 Swin Transformer 注意力 logits 中，因此不是完全无先验的纯无监督方法，而是先验增强的无监督方法。