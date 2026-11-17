# CS5491-RA-NSC

Residual-Aware Neural Spectral Capacity for Transformer Architecture Selection

CS5491 小组成员为 **WZH** 和 **ZDY**。项目研究在 FFN 子块中显式建模恒等捷径，能否改善 Transformer 架构排序和参数预算下的选型。

- [项目 Proposal](doc/proposal.pdf)
- [两人分工、交接和验收](doc/WORK_PLAN.md)
- [Starter 数据、指标和运行说明](nsc-starter/README.md)
- [修改记录](CHANGELOG.md)

## 开发集复现

在仓库根目录用 Python 3.9 或以上运行。以下命令使用 starter 的标准库实现。

```sh
python3 nsc-starter/check_starter.py
python3 nsc-starter/evaluate.py --split development --output development_results.json
```

目前 [student_score.py](nsc-starter/student_score.py) 的两个入口均返回原始 NSC。开发输出可用于验证评估流程。WZH 交付估计器并与 ZDY 冻结方法、配置和提交 SHA 后，由 ZDY 运行 final 评估并归档结果。
