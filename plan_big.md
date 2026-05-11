# 模型优化总体计划

## 0. 文档定位

本文档用于规划后续模型优化方向。当前前提是：模型训练、测试和结果保存流程已经可以跑通，因此本文档不再讨论工程流程是否完整，而是聚焦模型效果提升。

优化核心目标：

1. 提升融合图像的清晰度和自然性。
2. 提升 mask 决策图的边界质量和可解释性。
3. 验证语义/结构先验 M 是否真正改善注意力和 mask。
4. 将当前 pseudo cross-attention 升级为真实双图像交叉注意力。
5. 为后续细胞显微图像多聚焦融合做好先验、损失和数据适配。

总体优化主线：

先验 M 优化 -> 注意力机制优化 -> mask 决策优化 -> 损失函数优化 -> 数据与实验验证 -> 论文表达支撑

## 1. 语义/结构先验 M 优化

### 1.1 当前问题

当前模型已经有 `SemanticPriorM`，它结合低级聚焦差异和 DeepLabv3 语义置信度生成先验图 M，并注入 Swin 窗口注意力 logits。

但仍存在几个问题：

- DeepLabv3 基于自然图像训练，对细胞图像可能不稳定。
- 先验 M 是否真的与聚焦差异区域一致，需要实验验证。
- M 是否真正传导到 mask，还没有形成充分证据。
- 当前先验形式偏固定，缺少低级结构先验、语义先验之间的对比。

### 1.2 优化方向

优先优化先验 M 的可靠性，而不是一开始就堆叠新模块。

可优化点：

- DeepLabv3 输入方式。
- LoG 差异先验。
- Gradient 差异先验。
- Sobel 边缘先验。
- Local contrast 先验。
- 多尺度 focus measure 先验。
- 语义先验与低级结构先验的组合方式。

//

### 1.3 实施步骤

第一步：固定当前默认先验版本，作为 baseline。

第二步：测试关闭语义先验后的模型表现，观察 fused 和 mask 的退化程度。

第三步：测试只使用低级结构先验，不使用 DeepLabv3。

第四步：测试 DeepLabv3 语义先验与低级结构先验组合是否优于单独低级先验。

第五步：在细胞图像数据到位后，优先验证低级结构先验，再决定是否继续保留 DeepLabv3。

### 1.4 判断标准

先验优化是否有效，主要看：

- M 是否覆盖聚焦差异明显区域。
- M 是否与 mask 边界存在可解释关系。
- 关闭 M 后 mask 是否明显变差。
- 增强 M 后 fused 是否更清晰且伪影更少。
- 在细胞图像上，M 是否仍然稳定。

## 2. Swin 注意力先验注入机制优化

### 2.1 当前问题

当前模型将 M 作为 logits-level bias 注入窗口注意力，这个方向是合理的，但当前 bias 形式可能偏温和。

当前风险：

- M 对 attention 的实际影响可能不够明显。
- prior_beta 的最佳范围未知。
- 不同 bias 形式对 mask 和 fused 的影响尚未验证。

### 2.2 优化方向

重点优化 M 进入注意力 logits 的方式。

可优化点：

- prior_beta 固定值消融。
- prior_beta 可学习。
- sum bias。
- product bias。
- log bias。
- difference bias。
- hybrid bias。

### 2.3 实施步骤

第一步：做 prior_beta 消融，确认先验强度是否影响输出。

第二步：在默认 bias 形式下确定一个稳定 prior_beta 范围。

第三步：只选择一种新的 bias 形式进行测试，避免变量过多。

第四步：如果新 bias 有提升，再扩展到多种 bias 形式对比。

第五步：根据实验结果决定最终保留固定 prior_beta，还是使用 learnable_prior。

### 2.4 判断标准

注意力先验注入优化是否有效，主要看：

- mask 是否更贴合聚焦边界。
- fused 是否减少模糊过渡和局部错误选择。
- prior_beta 变化是否产生稳定趋势。
- bias 变强后是否引入过度关注、噪声或伪影。

## 3. Bottleneck 真实双图像 Cross-Attention 优化

### 3.1 当前问题

当前 bottleneck 的 `CrossAttentionFuse` 形式上使用了 query、key、value 注意力，但 token 来源并不是真正的 img1 与 img2 双分支。

当前实现更准确地说是 pseudo cross-attention 或 asymmetric token interaction。

主要问题：

- t1 和 t2 来自同一个融合 token 序列。
- t2 只是 token 顺序反转。
- 没有显式建模 img1 token 查询 img2 token 的过程。
- 论文中如果称为真实 cross-attention 不够严谨。

### 3.2 优化方向

将 bottleneck 升级为真实双图像交叉注意力。

可优化点：

- 分别从 s4_f1 和 s4_f2 生成 token。
- 建立 img1 -> img2 的注意力。
- 建立 img2 -> img1 的注意力。
- 融合双向 cross-attention 输出。
- 保留旧版本作为对照实验。

### 3.3 实施步骤

第一步：保留当前 pseudo cross-attention 作为 baseline。

第二步：设计真实双分支 token 输入方式。

第三步：实现双向 cross-attention。

第四步：保证输出通道和 decoder 输入保持一致。

第五步：对比旧版本和真实 cross-attention 版本。

第六步：观察 mask 是否减少单图偏向，fused 是否更稳定。

### 3.4 判断标准

真实 cross-attention 是否有效，主要看：

- mask 是否减少偏向 img1 或 img2。
- 大区域前景/背景选择是否更一致。
- fused 是否减少局部错选。
- 参数量和显存增长是否可接受。
- 与旧 pseudo cross-attention 对比是否有稳定收益。

## 3.5 掩码决策前的注意力机制优化

该阶段插入在瓶颈真实交叉注意力验证之后，掩码决策图优化之前。

原因：

- 真实交叉注意力瓶颈已验证有效。
- 在直接优化掩码决策图之前，应加强特征交互和注意力模块。
- 目的是为后续掩码解码器提供更清晰、更具判别性的特征，而不是让掩码分支来补偿弱特征交互。

优化范围：

- 通道注意力升级：
  采用ECA-Net风格的轻量级通道注意力，替代过于简单的通道加权模块。
- 空间注意力升级：
  添加空间注意力，使模型能在特征图级别强调聚焦区域和结构边界。
- 多尺度注意力：
  测试具有不同感受野的并行或阶段式注意力，特别是窗口大小为4、8、16等。
- 交叉/自注意力组合：
  对比纯自注意力、纯交叉注意力以及自到交叉混合注意力。
- 阶段级插入策略：
  在Stage1、Stage2-3、Stage4、Bottleneck和Decoder中分别测试注意力放置，再进行组合。

推荐消融顺序：

1. 保持已验证的`true_cross`瓶颈作为当前主要基线。
2. 首先仅添加ECA风格通道注意力。
3. 然后仅添加空间注意力。
4. 然后测试通道+空间注意力组合。
5. 然后在选定阶段测试多尺度窗口注意力。
6. 最后讨论在进入掩码优化前，解码器端的边界精化注意力是否仍需要。

设计约束：

- 此阶段不改变先验M的生成。
- 在测试注意力模块本身时，不改变先验偏置模式或prior_beta。
- 此阶段不改变损失函数或掩码监督。
- 保持`fused_image = mask * img1 + (1 - mask) * img2`不变。
- 每次运行应回答注意力优化是否改善了`fused`、`mask`以及先验M与掩码之间的关系。

人工评估重点：

- 融合图像细节是否更清晰。
- 错误源选择是否减少。
- 掩码是否在不改变直接掩码损失的情况下变得更稳定。
- 注意力是否引入伪影、晕圈、过度锐化或过高的计算成本。
- 收益是否足够稳定，以支撑新注意力基线进入掩码决策图优化。

## 4. Mask 决策图优化

### 4.1 当前问题

当前模型采用 mask 驱动融合，这个设计可解释性强，但融合质量高度依赖 mask。

当前可能存在的问题：

- mask 边界不贴合真实聚焦区域。
- mask 局部过平滑。
- mask 在纹理区域碎片化。
- mask 偏向某一张源图像。
- 聚焦/散焦边界处决策不稳定。

### 4.2 优化方向

mask 优化应同时从 decoder、loss 和 prior 对齐三个方向推进。

可优化点：

- mask bin loss 权重。
- mask TV loss 权重。
- prior-mask alignment loss。
- edge-aware mask loss。
- decoder full-resolution refinement。
- 将 M_full 引入 mask_head 前。
- 增加边界增强分支。

### 4.3 实施步骤

第一步：先通过可视化和统计明确 mask 的主要问题类型。

第二步：优先调节 mask bin 和 TV 权重，观察是否能改善。

第三步：加入 prior-mask alignment loss，让 M 与 mask 建立显式关系。

第四步：如果边界仍然差，再考虑 decoder refinement。

第五步：如果纹理区域碎片化严重，再考虑 edge-aware 或区域一致性约束。

### 4.4 判断标准

mask 优化是否有效，主要看：

- mask 是否更明确地区分清晰区域。
- mask 是否避免大面积灰度化。
- mask 是否减少孤立碎片。
- mask 边界是否更贴合聚焦/散焦切换区域。
- fused 是否因此更清晰、更自然。

## 5. 损失函数优化

### 5.1 当前问题

当前 `ImprovedFusionLoss` 已经包含 detail、SSIM、mask bin 和 mask TV，是合理的第一版无监督融合损失。

但当前损失仍有不足：

- 没有显式约束 M 与 mask 的一致性。
- 对细胞图像的边界、纹理和局部对比度不一定适配。
- mask TV 可能导致边界过平滑。
- detail loss 可能过度关注高频噪声或纹理。

### 5.2 优化方向

损失函数优化不应一次性加入太多项，而应围绕当前问题逐步增加。

可优化点：

- prior-mask alignment loss。
- edge-aware mask loss。
- multi-scale gradient loss。
- local contrast preservation loss。
- cell boundary loss。
- fluorescence intensity consistency loss。

### 5.3 实施步骤

第一步：调整现有 mask bin 和 TV 权重。

第二步：加入 prior-mask alignment loss，建立 M 与 mask 的闭环。

第三步：如果边界仍然不佳，再加入 edge-aware mask loss。

第四步：细胞图像阶段再考虑 local contrast 和 cell boundary 相关损失。

第五步：每新增一个 loss，都必须做开启/关闭消融。

### 5.4 判断标准

损失优化是否有效，主要看：

- loss 是否稳定下降。
- mask 是否更可靠。
- fused 是否更清晰。
- 是否减少伪影、重影和边界模糊。
- 新 loss 是否带来稳定收益，而不是只改善个别样本。

## 6. Decoder 与边界恢复优化

### 6.1 当前问题

当前 decoder 使用多尺度 guide feature 恢复 mask，但细粒度边界仍可能不足。

主要问题：

- 高层特征空间分辨率较低。
- s1_g、s2_g、s3_g 虽然提供浅层信息，但未必足够恢复锐利边界。
- mask_head 前缺少针对边界的 refinement。
- 已定义的部分上采样或解码模块可能没有发挥作用。

### 6.2 优化方向

decoder 优化应保持克制，优先使用小改动验证效果。

可优化点：

- full-resolution refinement block。
- M_full 与 decoder 特征融合。
- 浅层边缘特征补充。
- mask_head 前增加轻量边界增强。
- 使用更强的 skip feature 融合方式。

### 6.3 实施步骤

第一步：观察当前 mask 边界是否是主要瓶颈。

第二步：若确认是边界问题，优先测试 mask_head 前的轻量 refinement。

第三步：若 M_full 与边界一致性较高，再测试将 M_full 引入 decoder。

第四步：如果细胞图像边界复杂，再考虑专门边界分支。

### 6.4 判断标准

decoder 优化是否有效，主要看：

- mask 边界是否更锐利。
- fused 边界是否减少模糊和 halo。
- 是否引入过多噪声。
- 是否显著增加参数量和显存。

## 7. 数据与训练策略优化

### 7.1 当前问题

当前训练数据规模较小，主要适合结构调试和初步验证，不足以支撑泛化能力结论。

主要风险：

- 小数据过拟合。
- 合成模糊数据与真实多聚焦图像存在差异。
- 自然图像先验不一定适配细胞图像。
- 消融实验过多，变量容易混乱。

### 7.2 优化方向

数据优化应服务于模型优化，而不是简单增加数量。

可优化点：

- 明确训练、验证、测试拆分。
- 增加真实 Lytro/MFFW 测试。
- 增加真实细胞远近焦图像。
- 优化合成 defocus blur 方式。
- 构建困难样本集。
- 固定可复现实验配置。

### 7.3 实施步骤

第一步：整理当前已有数据来源和用途。

第二步：固定一组标准测试样本，长期用于可视化对比。

第三步：对每次结构或 loss 改动，都在同一测试集上对比。

第四步：细胞图像到位后，先做低级结构先验实验。

第五步：再决定是否引入 DeepLabv3、CellPose、SAM 或其他语义模型。

### 7.4 判断标准

数据和训练策略是否有效，主要看：

- 结果是否可复现。
- 消融实验是否变量清晰。
- 模型是否只在训练样本好看，还是在真实测试样本也稳定。
- 细胞图像上是否需要调整先验和 loss。

## 8. 实验与论文支撑优化

### 8.1 当前问题

当前模型已有多个潜在创新点，但需要实验支撑。

需要避免的问题：

- 只提出语义先验，但没有消融证明。
- 只展示 fused，不展示 M 和 mask。
- 把 pseudo cross-attention 误写成真实 cross-attention。
- 客观指标和主观图像缺少对应关系。

### 8.2 优化方向

实验设计要围绕论文贡献闭环展开。

核心实验：

- prior on/off。
- prior_beta 消融。
- semantic input 消融。
- semantic bias 形式消融。
- pseudo cross-attention 与 true cross-attention 对比。
- current loss 与 prior-mask alignment loss 对比。
- decoder refinement 对比。
- 自然图像与细胞图像迁移对比。

### 8.3 实施步骤

第一步：固定 baseline。

第二步：每个优化点只做单变量对比。

第三步：每个 run 保存 fused、mask、M、局部放大图和参数记录。

第四步：整理主观图、客观指标和失败案例。

第五步：根据实验结果更新论文表述。

### 8.4 判断标准

实验是否能支撑论文，主要看：

- 每个创新点是否有对应消融。
- 每个结论是否有图像和指标支撑。
- M、mask、fused 三者关系是否能解释清楚。
- 方法表述是否与实际代码一致。

## 9. 推荐优化优先级

### P0：当前最优先

1. 可视化 M、mask、fused，并建立诊断统计。
2. 做 prior on/off 和 prior_beta 消融。
3. 分析当前 mask 的主要问题。
4. 固定 baseline 和标准测试样本。

### P1：模型结构重点优化

1. 将 bottleneck pseudo cross-attention 升级为真实双分支 cross-attention。
2. 在 mask 决策图优化前，加入通道注意力、空间注意力和多尺度注意力机制优化。
3. 测试更有效的 semantic bias 形式。
4. 加入 prior-mask alignment loss。
5. 针对 mask 边界做 decoder 小改动。

### P2：细胞图像适配

1. 优先测试低级结构先验。
2. 判断 DeepLabv3 是否适合细胞图像。
3. 设计细胞图像专用 loss。
4. 增加下游任务或细胞边界评价。

### P3：论文与长期扩展

1. 整理消融实验表。
2. 整理主观对比图。
3. 整理客观指标。
4. 更新方法章节。
5. 暂缓 ONNX 和轻量化。

## 10. 建议的总体推进顺序

第一阶段：诊断与先验验证。

第二阶段：真实 cross-attention 结构升级。

第三阶段：注意力机制整体优化。

第四阶段：mask 与 loss 闭环优化。

第五阶段：decoder 边界恢复优化。

第六阶段：细胞图像先验与 loss 适配。

第七阶段：实验整理与论文表达。

整体原则：

不要同时改多个模块。每次优化只回答一个问题：这个改动是否让 M 更可靠、mask 更可信、fused 更清晰。


TODO(通道注意力机制太简单，需要修改：采用  """ECA-Net风格通道注意力"""
    空间注意力加入；多尺度注意力，同时考虑不同感受野的依赖关系
    )
TODO([**不同位置引入的组合策略**]

完整多尺度融合：

输入
│
├─ Stem
├─ ❶ Stage1
│  ├─ 显式交叉注意力（Self→Cross混合）
│  ├─ 多尺度注意力（窗口4,8,16并行）
│  └─ Channel + Spatial Attention
│
├─ ❷ Stage2-3
│  ├─ 自适应融合注意力（t1↔t2）
│  ├─ 多尺度Swin
│  └─ 通道+空间双重
│
├─ ❸ Stage4
│  ├─ 高分辨率显式交叉（在分支维）
│  └─ Deformable Attention（可选）
│
├─ ❹ Bottleneck
│  ├─ 三头交叉（q1-k2, q2-k1, 自融合）
│  └─ 上下文聚合
│
├─ ❺ Decoder
│  ├─ 多尺度融合上采样
│  └─ 边界精化注意力
│
└─ 输出

预期效果：+12-18%
成本：计算量+20-25%, 显存+15%

)
