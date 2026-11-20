# CS5491-RA-NSC

Residual-Aware Neural Spectral Capacity for Transformer Architecture Selection

CS5491 小组成员为 **WZH** 和 **ZDY**。项目研究在 FFN 子块中显式建模恒等捷径，能否改善 Transformer 架构排序和参数预算下的选型。

- [里程碑提交与复现入口（English）](Milestone/README.md) · [报告草稿（PDF）](Milestone/milestone_report.pdf)
- [开发集完整结果](Milestone/results/milestone_development.json) · [机制与数值检查](Milestone/results/milestone_mechanism.json)
- [项目 Proposal](doc/proposal.pdf)
- [两人分工、交接和验收](doc/WORK_PLAN.md)
- [Starter 数据、指标和运行说明](nsc-starter/README.md)
- [修改记录](CHANGELOG.md)

## 里程碑开发集结果

残差评分器使用相同的注意力项，针对各 FFN 形状做 32 次固定种子的成对 Monte Carlo 抽样。消融复用分支样本，只去掉恒等捷径。评分及消融仅接受架构配置，课程归档的既有验证集 PPL 用于评估。

| 开发集评分 | Spearman ρ | Kendall τ-b | 规模相近对 τ-b |
| --- | ---: | ---: | ---: |
| 原始 NSC | 0.9697 | 0.8614 | 0.6222 |
| 残差感知 | 0.9746 | 0.8844 | 0.6444 |
| 去捷径消融 | 0.9751 | 0.8853 | 0.6333 |

残差感知与消融在 8 组预算选型中的 regret 相同。两者均在总体排序上超过原始 NSC。完整结果还包含位置启发式、总参数、参考 FLOPs 及随机选型。

## 开发集复现

在仓库根目录运行。使用 Python 3.12（已验证 3.12.14）及 [requirements.txt](requirements.txt) 中固定版本的 NumPy。归档数据随仓库提供，实验在 CPU 上完成。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
replay_dir=$(mktemp -d)
python nsc-starter/check_starter.py
PYTHONPATH=nsc-starter python Milestone/check_residual.py --output "$replay_dir/milestone_mechanism.json"
python nsc-starter/evaluate.py --split development --output "$replay_dir/milestone_development.json"
printf '%s\n' "$replay_dir"
```

在 16 GB Apple M1 MacBook Pro（Python 3.12.14、NumPy 2.3.5、Accelerate）上，starter 自检耗时不足 1 秒，机制检查约 7 秒，100 个架构与 8 组预算的开发集评估约 2 分钟。开发 JSON 中的 `student` 对应残差感知方法，`ablation` 对应去捷径消融。两者共用 FFN 样本缓存，所以第二项计时反映缓存复用。以上命令将结果写到临时目录，可与 `Milestone/results/` 的归档数值核对。计时随机器和运行条件变化。

已有 PDF 可直接用于里程碑。重新生成需安装 ReportLab，并传入包含 Times New Roman 常规、粗体、斜体 TTF 文件的目录。以下以 macOS 的字体位置为例。

```sh
python -m pip install reportlab==4.4.9
python Milestone/build_milestone_report.py --font-dir '/System/Library/Fonts/Supplemental' \
  --output "$replay_dir/milestone_report.pdf"
```

本次里程碑仅评估开发集。WZH 与 ZDY 冻结方法、配置及提交 SHA 后，由 ZDY 运行最终评估并归档结果。
