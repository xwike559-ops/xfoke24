# 无语义先验 Baseline

## 1. 目的

本文档记录 `SemanticPriorM` 关闭后的对照实验设置。

该实验用于回答一个核心问题：

关闭先验 M 后，融合图像和 mask 是否明显退化。

## 2. 实验变量

本对照实验只改变一个变量：

- `use_semantic_prior`: False

其余训练参数、数据集、loss 权重、epoch、batch size 应尽量与默认完整 M baseline 保持一致。

## 3. 推荐实验命名

建议实验名使用：

- `prior_off`
- `no_semantic_prior`
- `baseline_no_prior`

如果需要和完整 M baseline 配对，建议使用同一组训练参数，例如：

- `m_default`
- `m_off`

## 4. 运行方式

在训练入口中使用 `--no-semantic-prior` 即可关闭 `SemanticPriorM`。

该设置会写入本次 run 的 `run_meta.json`，字段位于 `semantic_prior.use_semantic_prior`。

## 5. 需要观察的结果

重点观察：

1. fused 图像是否更模糊。
2. fused 图像是否出现更多局部错选。
3. mask 是否更碎片化。
4. mask 是否更偏向某一张输入图像。
5. mask 边界是否更不稳定。

## 6. 通过标准

该步骤不要求无先验模型更好。

它的目标是建立对照组，用来判断默认先验 M 是否有实际贡献。

如果关闭先验后结果明显退化，说明 M 可能有效。

如果关闭先验后结果几乎不变，说明 M 的生成或注入方式需要继续优化。
