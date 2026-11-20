# CS5491 project milestone

This project asks whether including the feed-forward network's identity shortcut in a training-free Transformer score improves architecture ranking or selection under a parameter cap. The milestone compares the residual-aware score with a shortcut ablation and the course baselines on 100 development architectures.

## Submission files

| Item | Location |
| --- | --- |
| English report draft, 3 pages | [milestone_report.pdf](milestone_report.pdf) |
| Development ranking and all 8 budget and top-k combinations | [milestone_development.json](results/milestone_development.json) |
| Mechanism checks and sampling-error measurements | [milestone_mechanism.json](results/milestone_mechanism.json) |
| Scorer and shortcut ablation | [student_score.py](../nsc-starter/student_score.py) |
| Benchmark evaluation and course data | [evaluate.py](../nsc-starter/evaluate.py), [data and provenance](../nsc-starter/data/provenance.json) |
| Numerical checks and report source | [check_residual.py](check_residual.py), [build_milestone_report.py](build_milestone_report.py) |

The Canvas submission consists of the PDF, the [GitHub repository URL](https://github.com/wzhnbsixsixsix/CS5491-RA-NSC), and the [`milestone` tag](https://github.com/wzhnbsixsixsix/CS5491-RA-NSC/tree/milestone). The [change log](../CHANGELOG.md) traces changes to the scoring code, experiments, and report.

## Reproduce the report results

From the repository root, use Python 3.12 and install the pinned NumPy version. The course archive is included under `nsc-starter/data/` and is checked against its supplied hashes. The commands use the archived validation perplexities and run on a CPU without model training.

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

On a 16 GB Apple M1 with Python 3.12.14, NumPy 2.3.5 and Accelerate, the starter checks took under a second, the mechanism checks about 7 seconds, and the 100-architecture evaluation about 2 minutes. Expect both checks to print `PASS`. In the development JSON, `student` is the residual-aware score and `ablation` removes the shortcut. Check `ranking` and `selection` against the archived development JSON and `comparisons` against the mechanism JSON. Allow for floating-point differences and machine-dependent timings. The second scoring pass reuses FFN samples cached by the first pass.

To regenerate the report and its tables from the archived results, install ReportLab and point it to the Times New Roman regular, bold, and italic TTF files. This example uses the macOS font path.

```sh
python -m pip install reportlab==4.4.9
python Milestone/build_milestone_report.py \
  --font-dir '/System/Library/Fonts/Supplemental' \
  --output "$replay_dir/milestone_report.pdf"
```

The milestone evaluates the development split. The final split is reserved for assessment after the method and evaluation settings are frozen.
