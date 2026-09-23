# results

Aggregated outputs from the evaluation harness. Small summaries committed to
git; large artifacts (figures, raw predictions) ignored.

Layout:

```
results/
├── method_comparison.csv           # the main Δ mAP table
├── false_positives.csv              # per-method FPR on empty frames
├── latency.csv                      # per-method, per-stage timing
└── figures/                         # bar plots, Δ mAP charts
```
