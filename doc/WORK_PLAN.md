# CS5491 两人分工

本计划按 [Proposal](proposal.pdf) 的 Method and assumptions、Data and evaluation、Feasibility and schedule 三部分安排。日期采用 Proposal 的里程碑。

| 负责人 | 代码与实验 | 报告与交付物 |
| --- | --- | --- |
| **WZH** | 在 [student_score.py](../nsc-starter/student_score.py) 实现残差评分与消融，负责估计器选择及数值验证。复用 [nsc.py](../nsc-starter/nsc.py) 的配置解析和注意力评分约定。 | 推导与假设、估计器参数和选择依据、零分支与弱分支检查、小矩阵采样参考、估计误差和耗时、层换序不变性。撰写方法及机制验证。 |
| **ZDY** | 用 [check_starter.py](../nsc-starter/check_starter.py) 和 [evaluate.py](../nsc-starter/evaluate.py) 复现基线、核对数据与指标、整合评分器并运行开发和 final 评估。 | 运行命令、环境与硬件、原始结果和汇总图表、归档数据局限。撰写数据、评估协议、结果及局限。 |
| **共同** | WZH 核对评估中的方法和消融解释，ZDY 核对评分接口及可复现性。两人确认待冻结的版本和开发选择。 | 合并讨论与结论，检查报告中的图表、配置、结果和版本可以相互追溯。 |

## 评分器交接

- WZH 提供 `score(config)` 和 `ablation(config)`，两者只接收架构配置，返回有限标量，分数越高表示预测表现越好。
- 令 `B = W2 D W1`，`C(A) = ln det(I + AᵀA)`。每层 FFN 的完整方法使用 `E[C(I+B)] - d ln 2`，消融使用 `E[C(B)]`。两者共用 Gaussian 权重标准差 0.02、Bernoulli(0.5) 门和归一化恒等近似。保持注意力评分一致，逐层相加。原始 NSC 单列为基线。
- WZH 交付可运行提交、参数与随机种子（如使用）、数值验证命令和结果。ZDY 用 `evaluate.py --split development` 验证整个 100 架构开发集能完成评分，保存结果并回传异常及比较表。两人根据开发结果审定估计器。

## 待办 · 原始 NSC + 捷径分

- [ ] **暂缓实施。** 保留原始 NSC 对注意力及两个 FFN 投影 `W1`、`W2` 的评分，再给每层加上恒等捷径的容量 `C(I) = d ln 2`。现有架构配置中各层的隐藏宽度均为 `d`，因此对 `L` 层模型，总分为 `original_nsc(config) + L × d × ln 2`。
- **比较目的。** 检查仅凭随宽度和深度增加的加分，能否重现 RA 在开发集上的部分收益。这个加分只取决于 `d` 和 `L`，因此宽度与层数都相同的架构之间，排序与原始 NSC 一致。它也提供了一个对照，用来比较单独给捷径加分与对 `I+B` 整体评分的效果。
- **后续验证。** 如启用该基线，在同一开发集上沿用现有协议，与原始 NSC、去捷径消融和 RA 比较整体及规模匹配排序，以及相同预算下的选型结果。

## 里程碑

### 11 月 19 日 · 开发集

里程碑材料于 2026-11-20 整理，见 [报告草稿](../Milestone/milestone_report.pdf)、[开发集结果](../Milestone/results/milestone_development.json) 和 [机制检查](../Milestone/results/milestone_mechanism.json)。两项评分已在开发集联调，最终集留待冻结后评估。

- **WZH** 交付候选估计器、消融和机制验证结果。
- **ZDY** 复现 starter 基线并完成集成开发集比较。记录对负 perplexity 的 Spearman、Kendall tau-b、并列和评分耗时，以及等隐藏宽度且总参数 min/max ≥ 0.90 的配对统计与配对数。
- **ZDY** 汇总 10M、20M、30M、50M 总参数上限下的 top-1 和 top-3 模拟选型。各方法使用相同可行池，报告最佳选中 PPL 和相对可行最优的 regret，纳入原始 NSC、位置启发式、总参数、参考 FLOPs、完整方法、消融和重复均匀随机选型。

### 12 月 3 日 · 最终交付

- **共同** 冻结估计器、开发选择、评估设置和 Git 提交 SHA。记录后再查看 final 结果。
- **ZDY** 在冻结版本运行 `python3 nsc-starter/evaluate.py --split final --output final_results.json`。归档原始 JSON、复现命令与结果表。默认结果文件被 Git 忽略，需将用于报告的快照放入明确命名的归档路径。
- **WZH** 检查完整方法与消融的解释，结合机制验证讨论排名和选型变化。**ZDY** 说明固定归档及每个架构单次记录对因果解释和训练随机性的限制。两人共同核对可复现提交和报告。
- 归档实验和数值验证构成核心交付。若考虑附加训练，WZH 先记录小型 decoder 试跑的硬件、吞吐与峰值内存，再由两人决定是否有资源做同设置、同数据划分、同参数预算的对照训练。

## 修改追溯

代码、实验或文档每次修改时，在 [CHANGELOG.md](../CHANGELOG.md) 记录日期、负责人、涉及文件、原因及验证结果。用于报告的实验记录同时保存运行命令、数据划分、提交 SHA 和输出位置。
