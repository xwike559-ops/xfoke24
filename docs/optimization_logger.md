模型优化日志

本文档记录每次模型优化步骤。每个条目都应保持变量变更清晰，以便后续的消融实验结果能回溯到代码、配置和视觉结论。

记录规则

1. 每个条目记录一个优化目标。
2. 除非条目中明确说明，否则保持基线不变。
3. 在进行实验之前，记录变更的文件和预期效果。
4. 测试后，记录确切的命令、运行目录和观察到的产物。
5. 将人工结论部分留给人类进行视觉评估。
6. 在检查完融合图像、掩码和先验M图之前，不要标记优化为已接受。

条目 001 - 底层先验模式扩展

日期：2026-05-09

状态：已实现，等待用户实验

设计对齐性：

• 遵循设计路线：先验M可靠性 -> 注意力引导 -> 掩码可靠性 -> 融合质量。

• 不改变融合公式。

• 此步骤不改变解码器、损失函数或瓶颈注意力。

• 保留现有的DeepLabv3语义先验路径，因为其语义可靠性已得到验证。

优化目标：

扩展SemanticPriorM使用的底层结构/焦点先验选项。这可以在保持相同语义增强分支的同时，对不同先验图进行受控的消融研究。

变更文件：

• src/models/hybrid_fusion_net.py

• src/main.py

• docs/optimization_logger.md

代码变更：

• 保留默认的prior_diff_mode=lap作为当前基线。

• 添加sobel作为现有梯度幅度先验的明确别名。

• 添加基于局部方差/对比度差异的local_contrast先验。

• 添加基于多个窗口大小下平均局部对比度和梯度响应的multiscale_focus先验。

• 扩展了--prior-diff-mode的CLI选项。

可用先验模式：
lap
grad
sobel
local_contrast
multiscale_focus


推荐的消融顺序：

1. lap：当前默认基线。
2. grad：现有的梯度差异基线。
3. sobel：为论文/实验清晰度而命名的显式Sobel。
4. local_contrast：更强的局部纹理和细胞结构对比度先验。
5. multiscale_focus：跨纹理尺度的更稳健的焦点测量先验。

建议命令：
python start.py --experiment-name prior_lap --prior-diff-mode lap
python start.py --experiment-name prior_grad --prior-diff-mode grad
python start.py --experiment-name prior_sobel --prior-diff-mode sobel
python start.py --experiment-name prior_local_contrast --prior-diff-mode local_contrast
python start.py --experiment-name prior_multiscale_focus --prior-diff-mode multiscale_focus


预期观察结果：

• lap：对尖锐的高频差异敏感；是物体边缘的良好基线。

• grad/sobel：应更直接地突出连续边缘过渡。

• local_contrast：可能有助于纹理或微观结构，但也可能放大噪声。

• multiscale_focus：可能在混合大小的焦点结构上更稳定，但存在先验边界更平滑的风险。

验证清单：

• 前向传播适用于每个先验模式。

• fused_image、mask和mask_logits保持与之前相同的形状。

• 训练元数据记录semantic_prior.prior_diff_mode。

• 每次运行在experiments/runs目录下产生可比较的融合/掩码输出。

手动测试结论：
运行目录：

视觉结论：

掩码结论：

先验M结论：

接受的模式：

观察到的问題：

下一步：

## Entry 002 - Low-Level Prior And Combination Verification

Date: 2026-05-09

Status: user verified, detailed visual conclusions pending manual fill-in

Design Alignment:

- This entry belongs to the prior M optimization stage in `plan_big.md`.
- The purpose is to compare how different low-level priors affect `M`, `mask`, and `fused_image`.
- The user has verified that the semantic model is reliable and helpful for fusion quality.
- Therefore, the next optimization should not remove DeepLabv3, but should study how semantic prior and low-level structural prior should be combined.

Verified Items:

- `lap` low-level structure prior.
- `grad` / `sobel` edge prior.
- `local_contrast` local contrast prior.
- `multiscale_focus` multi-scale focus measure prior.
- Semantic prior and low-level structure prior combination behavior.

Manual Test Conclusion:

```text
Run directories:

Best low-level prior:

Best semantic/structure combination:

Fused-image observation:

Mask observation:

Prior-M observation:

Main problems:

Initial decision:

Next discussion:
Semantic prior and low-level structure prior combination strategy.
```

## Entry 003 - Semantic And Low-Level Prior Combination Modes

Date: 2026-05-09

Status: implemented, pending user experiments

Design Alignment:

- This step follows the design route: improve prior M before changing attention, mask decoder, or loss.
- The default behavior remains `prior_semantic_mode=edge`, so the existing baseline is preserved.
- This step only changes how semantic prior and low-level structural prior are combined inside `SemanticPriorM`.
- The fusion formula remains unchanged: `fused_image = mask * img1 + (1 - mask) * img2`.

Optimization Target:

Add four semantic/structure prior combination strategies for controlled ablation.

Changed Files:

- `src/models/hybrid_fusion_net.py`
- `src/main.py`
- `docs/optimization_logger.md`

Available Combination Modes:

```text
edge
gate
weighted_sum
adaptive
```

Combination Formulas:

```text
D = normalized low-level focus/structure difference
S_conf = DeepLabv3 confidence from avg(img1, img2)
S_edge = normalized gradient magnitude of S_conf
```

```text
edge:
M = Normalize(D * (1 + lambda * S_edge)) ^ gamma
```

```text
gate:
M = Normalize(D * S_conf) ^ gamma
```

```text
weighted_sum:
M = Normalize(alpha * D + (1 - alpha) * S_edge) ^ gamma
```

```text
adaptive:
M = Normalize(D * (1 + lambda * S_edge) + mu * D * S_conf) ^ gamma
```

New CLI Parameters:

```text
--prior-semantic-mode edge|gate|weighted_sum|adaptive
--prior-semantic-alpha 0.7
--prior-semantic-mu 0.5
```

Recommended Ablation Order:

1. `edge`: current default baseline.
2. `gate`: tests whether semantic confidence suppresses noisy low-level responses.
3. `weighted_sum`: tests whether semantic edge should independently contribute to M.
4. `adaptive`: tests semantic edge enhancement plus semantic confidence gating while keeping D as the core.

Suggested Commands:

```powershell
python src/main.py --experiment-name semantic_edge --prior-semantic-mode edge
python src/main.py --experiment-name semantic_gate --prior-semantic-mode gate
python src/main.py --experiment-name semantic_weighted_sum --prior-semantic-mode weighted_sum --prior-semantic-alpha 0.7
python src/main.py --experiment-name semantic_adaptive --prior-semantic-mode adaptive --prior-semantic-mu 0.5
```

Manual Test Conclusion:

```text
Run directories:

Best semantic combination mode:

Best alpha:

Best mu:

Fused-image observation:

Mask observation:

Prior-M observation:

Main problems:

Initial decision:

Next step:
```

## Entry 004 - Swin Attention Prior Bias Modes

Date: 2026-05-09

Status: implemented, pending user experiments

Design Alignment:

- This entry starts the second optimization direction in `plan_big.md`: Swin attention prior injection mechanism.
- The prior map M generation is unchanged in this step.
- The decoder, loss, bottleneck interaction, and fusion formula are unchanged.
- The default mode is `sum_log`, preserving the previous implementation as baseline.

Optimization Target:

Make the logits-level semantic prior injection in Swin window attention configurable.

Changed Files:

- `src/models/swin2d_parts.py`
- `src/models/hybrid_fusion_net.py`
- `src/main.py`
- `docs/optimization_logger.md`

Available Prior Bias Modes:

```text
sum_log
sum
product
diff
hybrid
```

Bias Formulas:

```text
m_i, m_j = prior values of two tokens in the same window
sum_bias = 0.5 * (m_i + m_j)
```

```text
sum_log:
bias_ij = log(sum_bias + eps)
```

```text
sum:
bias_ij = sum_bias
```

```text
product:
bias_ij = m_i * m_j
```

```text
diff:
bias_ij = -abs(m_i - m_j)
```

```text
hybrid:
bias_ij = log(sum_bias + eps) + m_i * m_j - abs(m_i - m_j)
```

New CLI Parameter:

```text
--prior-bias-mode sum_log|sum|product|diff|hybrid
```

Recommended Ablation Order:

1. `sum_log`: current baseline.
2. `sum`: tests whether the log sharpening is necessary.
3. `product`: emphasizes pairs where both tokens have high prior response.
4. `diff`: encourages attention between tokens with similar prior response.
5. `hybrid`: combines current log sharpening, high-high preference, and similarity preference.

Suggested Commands:

```powershell
python src/main.py --experiment-name bias_sum_log --prior-bias-mode sum_log
python src/main.py --experiment-name bias_sum --prior-bias-mode sum
python src/main.py --experiment-name bias_product --prior-bias-mode product
python src/main.py --experiment-name bias_diff --prior-bias-mode diff
python src/main.py --experiment-name bias_hybrid --prior-bias-mode hybrid
```

Manual Test Conclusion:

```text
Run directories:

Best prior_bias_mode:

Fused-image observation:

Mask observation:

Prior-M and mask relation:

Main problems:

Next step:
prior_beta ablation after prior_bias_mode is selected.
```
