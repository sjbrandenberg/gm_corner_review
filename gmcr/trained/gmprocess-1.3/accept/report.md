# accept/reject model comparison

141992 reviewed traces, 277 events, accept rate 0.893

| model | AUC | Brier | log loss |
|---|---|---|---|
| baseline | 0.471 | 0.0955 | 0.3401 |
| simple | 0.884 | 0.0675 | 0.2300 |
| lightgbm | 0.942 | 0.0506 | 0.1710 |

Accepted records lost when flagging a share of the analysts' rejections:

- baseline, flag_50pct_of_rejections: 60.4% of accepted records flagged
- baseline, flag_80pct_of_rejections: 100.0% of accepted records flagged
- baseline, flag_90pct_of_rejections: 100.0% of accepted records flagged
- simple, flag_50pct_of_rejections: 5.2% of accepted records flagged
- simple, flag_80pct_of_rejections: 19.3% of accepted records flagged
- simple, flag_90pct_of_rejections: 31.3% of accepted records flagged
- lightgbm, flag_50pct_of_rejections: 1.8% of accepted records flagged
- lightgbm, flag_80pct_of_rejections: 8.2% of accepted records flagged
- lightgbm, flag_90pct_of_rejections: 15.3% of accepted records flagged

Comparisons (log loss difference, event bootstrap 95% CI):

- baseline_vs_simple: +0.1101 [+0.0991, +0.1218]
- simple_vs_lightgbm: +0.0590 [+0.0509, +0.0670]

By analyst (AUC / log loss):

| analyst | events | traces | accept rate | simple | lightgbm |
|---|---|---|---|---|---|
| 3 | 152 | 75037 | 0.902 | 0.899 / 0.211 | 0.950 / 0.159 |
| 9 | 47 | 22812 | 0.838 | 0.876 / 0.338 | 0.932 / 0.269 |
| 7 | 30 | 21978 | 0.927 | 0.914 / 0.171 | 0.963 / 0.124 |
| 14 | 14 | 13546 | 0.914 | 0.911 / 0.188 | 0.962 / 0.131 |
| unknown | 24 | 5521 | 0.780 | 0.873 / 0.416 | 0.926 / 0.257 |
| 5 | 8 | 1996 | 0.979 | 0.775 / 0.124 | 0.920 / 0.067 |
| 40 | 1 | 688 | 0.839 | 0.837 / 0.339 | 0.893 / 0.295 |
| 4 | 1 | 414 | 0.993 | 0.962 / 0.050 | 1.000 / 0.028 |

Top LightGBM predictors (share of total gain):

- ln_pga_ratio: 48.2%
- ln_snr10_fhi: 6.9%
- ln_stress_drop: 5.3%
- depth_km: 5.1%
- magnitude: 4.1%
- ln_fchp_gui: 3.6%
- zero_crossing_rate: 2.6%
- ln_snr_at_3hz: 2.5%
- ln_rrup: 1.9%
- ln_signal_duration: 1.7%
- rx: 1.4%
- psa_rough_short: 1.3%

Recommendation: **lightgbm**
