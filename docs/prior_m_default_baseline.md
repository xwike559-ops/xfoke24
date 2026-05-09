# 默认语义/结构先验 M Baseline

## 1. 目的

本文档固定当前默认 `SemanticPriorM` 作为后续先验优化实验的 baseline。

后续所有关于 M 的优化都需要与该 baseline 对比，避免同时改变多个变量。

## 2. 当前默认先验组成

默认 M 由两部分组成：

1. 低级聚焦差异：使用 Laplacian 响应差异。
2. 语义边界增强：使用 DeepLabv3 语义置信图的梯度边缘。

当前默认逻辑可以概括为：

- 先计算 img1 与 img2 的灰度 Laplacian 差异，得到聚焦差异图 D。
- 再将 img1 与 img2 的平均图输入 DeepLabv3，得到语义置信图 S_conf。
- 对 S_conf 求梯度，得到语义边界响应 S_edge。
- 用 S_edge 增强 D，得到先验图 M。
- 对 M 做归一化和 gamma 调整。

## 3. 当前默认参数

当前默认参数如下：

- `use_semantic_prior`: True
- `prior_diff_mode`: lap
- `prior_semantic_mode`: edge
- `prior_lam`: 2.0
- `prior_gamma`: 1.5
- `use_deeplab`: True
- `prior_beta`: 1.0
- `learnable_prior`: False

其中 `prior_beta` 不属于 M 的生成参数，而是控制 M 注入 Swin 注意力 logits 的强度。为了保持 baseline 清晰，默认先保持 `prior_beta = 1.0`。

## 4. 当前 baseline 的意义

该 baseline 代表当前模型的完整先验版本：

- 有低级聚焦差异。
- 有 DeepLabv3 语义置信度。
- 有语义边界增强。
- 有多尺度注入 Swin 注意力。
- 有固定强度的先验注入。

后续实验需要证明：

- 关闭先验后模型是否退化。
- 只使用低级结构先验是否足够。
- DeepLabv3 语义边界是否带来额外收益。
- prior_beta 改变是否稳定影响 mask 和 fused。

## 5. 后续对照组

建议后续按以下顺序建立对照：

1. 当前默认完整 M baseline。
2. 关闭语义先验。
3. 保留 M 生成但让 prior_beta 为 0。
4. 只使用低级结构先验。
5. 使用 DeepLabv3 与低级结构先验组合。

## 6. 当前注意事项

Readme 中仍有部分旧描述写到 DeepLabv3 使用 img1 作为语义输入。当前代码实际使用 img1 和 img2 的平均图作为语义输入。

后续如果重新整理论文或 Readme，应将这一点同步更新，避免描述和代码不一致。
