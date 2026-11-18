# fclp model comparison

126826 traces, 277 events, 5-fold event-grouped CV

| model | RMSE ln | bias ln | within 20% | within x1.5 |
|---|---|---|---|---|
| baseline_gmprocess | 0.159 | +0.020 | 0.96 | 0.98 |
| simple | 0.135 | -0.003 | 0.95 | 0.98 |
| lightgbm | 0.115 | -0.001 | 0.96 | 0.98 |

Comparisons (RMSE difference, ln units, event bootstrap 95% CI):

- baseline_gmprocess_vs_simple: +0.0237 [+0.0192, +0.0278], relative 14.9%
- baseline_gmprocess_vs_lightgbm: +0.0434 [+0.0354, +0.0513], relative 27.4%
- simple_vs_lightgbm: +0.0197 [+0.0156, +0.0241], relative 14.6%

RMSE (ln) by analyst:

| analyst | events | traces | baseline_gmprocess | simple | lightgbm |
|---|---|---|---|---|---|
| 3 | 152 | 67698 | 0.167 | 0.133 | 0.103 |
| 7 | 30 | 20385 | 0.169 | 0.148 | 0.131 |
| 9 | 47 | 19116 | 0.159 | 0.146 | 0.133 |
| 14 | 14 | 12379 | 0.004 | 0.071 | 0.100 |
| unknown | 24 | 4306 | 0.217 | 0.198 | 0.172 |
| 5 | 8 | 1954 | 0.047 | 0.058 | 0.054 |
| 40 | 1 | 577 | 0.183 | 0.169 | 0.172 |
| 4 | 1 | 411 | 0.040 | 0.046 | 0.045 |

Best simple: simple; best ML: lightgbm.
Recommendation: **lightgbm** (ML needs >= 10% lower RMSE than the best simple model, with a CI excluding 0)
