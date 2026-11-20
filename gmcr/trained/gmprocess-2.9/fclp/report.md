# fclp model comparison

121803 traces, 277 events, 5-fold event-grouped CV; without magnitude, depth_km

| model | RMSE ln | bias ln | within 20% | within x1.5 |
|---|---|---|---|---|
| baseline_gmprocess | 0.240 | +0.085 | 0.74 | 0.92 |
| simple | 0.218 | -0.027 | 0.81 | 0.94 |
| lightgbm | 0.182 | -0.027 | 0.87 | 0.96 |

Comparisons (RMSE difference, ln units, event bootstrap 95% CI):

- baseline_gmprocess_vs_simple: +0.0225 [+0.0153, +0.0300], relative 9.4%
- baseline_gmprocess_vs_lightgbm: +0.0584 [+0.0510, +0.0661], relative 24.3%
- simple_vs_lightgbm: +0.0358 [+0.0305, +0.0421], relative 16.5%

RMSE (ln) by analyst:

| analyst | events | traces | baseline_gmprocess | simple | lightgbm |
|---|---|---|---|---|---|
| 3 | 152 | 65091 | 0.217 | 0.192 | 0.164 |
| 7 | 30 | 19697 | 0.244 | 0.225 | 0.179 |
| 9 | 47 | 18233 | 0.286 | 0.257 | 0.207 |
| 14 | 14 | 11722 | 0.251 | 0.241 | 0.221 |
| unknown | 24 | 4194 | 0.296 | 0.244 | 0.212 |
| 5 | 8 | 1925 | 0.171 | 0.290 | 0.179 |
| 40 | 1 | 552 | 0.469 | 0.340 | 0.285 |
| 4 | 1 | 389 | 0.105 | 0.141 | 0.086 |

Best simple: simple; best ML: lightgbm.
Recommendation: **lightgbm** (ML needs >= 10% lower RMSE than the best simple model, with a CI excluding 0)
