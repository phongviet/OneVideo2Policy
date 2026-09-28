# CVPR-style project report

`report.pdf` is the five-page hardware-free reproduction report. It uses the
official CVPR 2026 `cvpr.sty` and `ieeenat_fullname.bst` files.

The measured tables and plots are derived from `experiment-results.json`. To
re-run the added experiments and regenerate their artifacts:

```bash
.venv/bin/python scripts/run_report_experiments.py \
  --episodes 50 --training-seeds 42 43 44 45 46
```

Raw checkpoints, logs, and episode traces are written under the ignored
`results/report_experiments/` directory. The runner resumes completed rows and
updates the committed manifest, aggregate, plots, and LaTeX macros.

Compile the report from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```
