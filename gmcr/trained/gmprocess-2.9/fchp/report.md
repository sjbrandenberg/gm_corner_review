# fchp model comparison

121087 traces, 277 events, 5-fold event-grouped CV; without magnitude, depth_km

Analyst kept the gmprocess value for 1.0% of these traces.

| model | RMSE ln | bias ln | within 20% | within x1.5 |
|---|---|---|---|---|
| baseline_gmprocess | 0.476 | -0.123 | 0.50 | 0.73 |
| baseline_gui | 1.377 | -0.788 | 0.19 | 0.34 |
| simple | 0.346 | -0.000 | 0.48 | 0.81 |
| simple_2stage_soft | 0.346 | -0.000 | 0.48 | 0.81 |
| simple_2stage_hard | 0.346 | -0.000 | 0.48 | 0.81 |
| lightgbm | 0.290 | +0.007 | 0.59 | 0.86 |
| lightgbm_2stage_soft | 0.288 | +0.006 | 0.59 | 0.87 |
| lightgbm_2stage_hard | 0.288 | +0.007 | 0.59 | 0.87 |

Keep-gmprocess classifier (simple): AUC 0.663, Brier 0.0097

Keep-gmprocess classifier (lightgbm): AUC 0.851, Brier 0.0092

Comparisons (RMSE difference, ln units, event bootstrap 95% CI):

- baseline_gmprocess_vs_baseline_gui: -0.9013 [-0.9519, -0.8564], relative -189.4%
- baseline_gmprocess_vs_simple: +0.1301 [+0.1192, +0.1407], relative 27.3%
- baseline_gmprocess_vs_simple_2stage_soft: +0.1304 [+0.1196, +0.1410], relative 27.4%
- baseline_gmprocess_vs_simple_2stage_hard: +0.1301 [+0.1192, +0.1408], relative 27.3%
- baseline_gmprocess_vs_lightgbm: +0.1861 [+0.1727, +0.1999], relative 39.1%
- baseline_gmprocess_vs_lightgbm_2stage_soft: +0.1878 [+0.1747, +0.2017], relative 39.5%
- baseline_gmprocess_vs_lightgbm_2stage_hard: +0.1877 [+0.1747, +0.2016], relative 39.4%
- simple_2stage_soft_vs_lightgbm_2stage_soft: +0.0574 [+0.0508, +0.0643], relative 16.6%

RMSE (ln) by analyst:

| analyst | events | traces | baseline_gmprocess | simple_2stage_soft | lightgbm_2stage_soft |
|---|---|---|---|---|---|
| 3 | 152 | 65091 | 0.455 | 0.327 | 0.273 |
| 7 | 30 | 19697 | 0.414 | 0.339 | 0.282 |
| 9 | 47 | 18233 | 0.566 | 0.392 | 0.335 |
| 14 | 14 | 11006 | 0.459 | 0.350 | 0.276 |
| unknown | 24 | 4194 | 0.624 | 0.419 | 0.335 |
| 5 | 8 | 1925 | 0.492 | 0.349 | 0.309 |
| 40 | 1 | 552 | 0.668 | 0.438 | 0.399 |
| 4 | 1 | 389 | 0.358 | 0.269 | 0.224 |

Top LightGBM predictors (share of total gain):

- ln_snr2_flo: 30.9%
- ln_fchp_gmp: 17.6%
- ln_fchp_gui: 14.3%
- ln_snr3_flo: 9.6%
- pre_disp_ratio_003: 2.9%
- ln_snr5_flo: 1.5%
- ln_signal_duration: 1.2%
- poly_ratio_005: 1.2%
- ln_rrup: 1.0%
- poly_ratio_006: 1.0%
- snr_023: 0.9%
- pre_disp_ratio_007: 0.8%

Best simple: simple_2stage_soft; best ML: lightgbm_2stage_soft.
Recommendation: **lightgbm_2stage_soft** (ML needs >= 10% lower RMSE than the best simple model, with a CI excluding 0)
