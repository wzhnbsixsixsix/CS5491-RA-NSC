# 修改记录

## 2026-11-21 · 阶段报告的 Canvas 提交入口

- `Milestone/build_milestone_report.py` 和生成的 `Milestone/milestone_report.pdf` 在第一页加入可见、可点击的 GitHub `milestone` 链接，并标明评分与消融、评估及机制检查的代码入口。报告同时链接运行说明、数据来源、生成脚本和两份结果归档。
- `Milestone/README.md` 记录从 Canvas 上传的 PDF 如何找到上述文件。报告继续使用原有三页版式，代码和实验结果链接指向已发布的 `milestone` 标签。

## 2026-11-20 · 项目里程碑

- 方法与代码（WZH 原定负责范围）。`nsc-starter/student_score.py` 实现 32 次固定种子、按形状缓存的成对 Monte Carlo 残差估计与去捷径消融。`nsc-starter/nsc.py` 提取共用注意力项，保持原始 NSC 计算口径。`requirements.txt` 固定 NumPy 版本。
- 实验（ZDY 原定负责范围）。`Milestone/check_residual.py` 增加零分支、弱分支、SVD 对照、层换序、配置形式一致性及采样误差检查。归档 `Milestone/results/milestone_mechanism.json` 和 `Milestone/results/milestone_development.json`，覆盖 100 个开发架构和 8 种预算与候选数量组合。`nsc-starter/check_starter.py` 与 `nsc-starter/evaluate.py` 补充入口和数据读取说明。独立复跑自检、机制检查和开发集评估，归档数值在浮点容差内一致，最终集留待方法冻结后评估。
- 报告与交接。`Milestone/milestone_report.pdf` 新增课程格式的英文报告草稿，`Milestone/build_milestone_report.py` 根据同目录归档生成表格和 PDF。`Milestone/README.md` 提供提交版本、文件入口和英文复现说明。`README.md`、`nsc-starter/README.md` 和 `doc/WORK_PLAN.md` 更新运行环境、命令、预期耗时、结果与当前进度。发布版本和归档由 `milestone` 标签追溯。

## 2026-11-17 · GitHub 仓库更名

- GitHub 仓库更名为 `CS5491-RA-NSC`，并已同步本地 `origin` 地址。
- `README.md` 采用新的仓库名并保留项目全称。
- `README.md` 和 `doc/WORK_PLAN.md` 修正 Proposal 链接，对应远端已更新的 `doc/proposal.pdf`。

## 2026-11-17 · Proposal 分工

- `README.md` 增加仓库入口、Proposal、分工、修改记录和开发集复现命令。
- `doc/WORK_PLAN.md` 记录 WZH 负责残差评分与数值验证，ZDY 负责基线与归档评估，以及交接、开发集和 final 验收条件。
- `nsc-starter/student_score.py` 在模块说明中标出评分接口的负责人和交接文档；当前评分仍为 starter 基线。
- 验证通过 starter 自检及 100 个架构的开发集评估，覆盖 8 种预算与 k 组合。未运行 final。
