# accept/reject model comparison

135623 reviewed traces, 277 events, accept rate 0.898; without magnitude, depth_km

| model | AUC | Brier | log loss |
|---|---|---|---|
| baseline | 0.468 | 0.0916 | 0.3296 |
| simple | 0.897 | 0.0633 | 0.2163 |
| lightgbm | 0.952 | 0.0446 | 0.1526 |

Accepted records lost when flagging a share of the analysts' rejections:

- baseline, flag_50pct_of_rejections: 60.4% of accepted records flagged
- baseline, flag_80pct_of_rejections: 100.0% of accepted records flagged
- baseline, flag_90pct_of_rejections: 100.0% of accepted records flagged
- simple, flag_50pct_of_rejections: 4.7% of accepted records flagged
- simple, flag_80pct_of_rejections: 16.3% of accepted records flagged
- simple, flag_90pct_of_rejections: 27.0% of accepted records flagged
- lightgbm, flag_50pct_of_rejections: 1.3% of accepted records flagged
- lightgbm, flag_80pct_of_rejections: 6.3% of accepted records flagged
- lightgbm, flag_90pct_of_rejections: 12.4% of accepted records flagged

Comparisons (log loss difference, event bootstrap 95% CI):

- baseline_vs_simple: +0.1132 [+0.1023, +0.1247]
- simple_vs_lightgbm: +0.0637 [+0.0564, +0.0708]

By analyst (AUC / log loss):

| analyst | events | traces | accept rate | simple | lightgbm |
|---|---|---|---|---|---|
| 3 | 152 | 71747 | 0.907 | 0.910 / 0.198 | 0.957 / 0.141 |
| 9 | 47 | 21614 | 0.844 | 0.890 / 0.317 | 0.945 / 0.238 |
| 7 | 30 | 21150 | 0.931 | 0.921 / 0.163 | 0.967 / 0.110 |
| 14 | 14 | 12797 | 0.916 | 0.923 / 0.177 | 0.966 / 0.124 |
| unknown | 24 | 5327 | 0.787 | 0.888 / 0.400 | 0.934 / 0.244 |
| 5 | 8 | 1960 | 0.982 | 0.801 / 0.125 | 0.963 / 0.052 |
| 40 | 1 | 636 | 0.868 | 0.892 / 0.270 | 0.939 / 0.196 |
| 4 | 1 | 392 | 0.992 | 0.932 / 0.049 | 1.000 / 0.020 |

Top LightGBM predictors (share of total gain):

- ln_pga_ratio: 31.7%
- pre_disp_ratio_024: 13.1%
- pre_disp_ratio_023: 6.9%
- ln_signal_duration: 3.2%
- zero_crossing_rate: 2.5%
- rx: 2.1%
- ry0: 1.4%
- preevent_noise_s: 1.0%
- clipping_prob: 0.9%
- ln_rrup: 0.9%
- ln_stress_drop: 0.8%
- fas_noise_078: 0.7%

Recommendation: **lightgbm**
