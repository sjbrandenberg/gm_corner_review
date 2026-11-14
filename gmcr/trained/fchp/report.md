# fchp model comparison

126086 traces, 277 events, 5-fold event-grouped CV

Analyst kept the gmprocess value for 66.1% of these traces.

| model | RMSE ln | bias ln | within 20% | within x1.5 |
|---|---|---|---|---|
| baseline_gmprocess | 0.342 | -0.125 | 0.72 | 0.84 |
| baseline_gui | 1.373 | -0.772 | 0.19 | 0.35 |
| simple | 0.261 | +0.000 | 0.64 | 0.91 |
| simple_2stage_soft | 0.254 | +0.000 | 0.72 | 0.90 |
| simple_2stage_hard | 0.288 | -0.010 | 0.72 | 0.87 |
| lightgbm | 0.231 | +0.003 | 0.74 | 0.92 |
| lightgbm_2stage_soft | 0.229 | +0.001 | 0.76 | 0.92 |
| lightgbm_2stage_hard | 0.259 | +0.004 | 0.75 | 0.90 |

Keep-gmprocess classifier (simple): AUC 0.753, Brier 0.1809

Keep-gmprocess classifier (lightgbm): AUC 0.831, Brier 0.1516

Comparisons (RMSE difference, ln units, event bootstrap 95% CI):

- baseline_gmprocess_vs_baseline_gui: -1.0313 [-1.0882, -0.9811], relative -301.9%
- baseline_gmprocess_vs_simple: +0.0808 [+0.0709, +0.0900], relative 23.6%
- baseline_gmprocess_vs_simple_2stage_soft: +0.0879 [+0.0768, +0.0983], relative 25.7%
- baseline_gmprocess_vs_simple_2stage_hard: +0.0533 [+0.0392, +0.0663], relative 15.6%
- baseline_gmprocess_vs_lightgbm: +0.1103 [+0.0958, +0.1236], relative 32.3%
- baseline_gmprocess_vs_lightgbm_2stage_soft: +0.1128 [+0.0990, +0.1256], relative 33.0%
- baseline_gmprocess_vs_lightgbm_2stage_hard: +0.0827 [+0.0666, +0.0977], relative 24.2%
- simple_2stage_soft_vs_lightgbm_2stage_soft: +0.0249 [+0.0186, +0.0315], relative 9.8%

RMSE (ln) by analyst:

| analyst | events | traces | baseline_gmprocess | simple_2stage_soft | lightgbm_2stage_soft |
|---|---|---|---|---|---|
| 3 | 152 | 67698 | 0.313 | 0.240 | 0.211 |
| 7 | 30 | 20385 | 0.322 | 0.236 | 0.222 |
| 9 | 47 | 19116 | 0.438 | 0.303 | 0.288 |
| 14 | 14 | 11639 | 0.269 | 0.213 | 0.191 |
| unknown | 24 | 4306 | 0.468 | 0.318 | 0.275 |
| 5 | 8 | 1954 | 0.319 | 0.344 | 0.286 |
| 40 | 1 | 577 | 0.725 | 0.415 | 0.378 |
| 4 | 1 | 411 | 0.291 | 0.208 | 0.180 |

Top LightGBM predictors (share of total gain):

- ln_fchp_gmp: 70.1%
- ln_snr2_flo: 13.8%
- ln_fchp_gui: 6.7%
- pre_ratio_at_gmp: 1.8%
- magnitude: 1.4%
- depth_km: 0.9%
- poly_ratio_at_gmp: 0.6%
- ln_signal_duration: 0.6%
- ln_snr3_flo: 0.6%
- ln_rrup: 0.6%
- pre_ratio_at_gui: 0.4%
- clipping_prob: 0.4%

Best simple: simple_2stage_soft; best ML: lightgbm_2stage_soft.
Recommendation: **simple_2stage_soft** (ML needs >= 10% lower RMSE than the best simple model, with a CI excluding 0)
