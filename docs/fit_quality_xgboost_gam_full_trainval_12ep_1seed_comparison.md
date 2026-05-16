# Fit-quality comparison: `MI_correct` vs. `vif_only_no_collision`

Date: 2026-05-15

## Scope

This report compares the out-of-fold fit quality of the XGBoost and GAM meta-models for:

- `full_trainval_12ep_1seed_MI_correct`
- `full_trainval_12ep_1seed_vif_only_no_collision`

The assessment uses the exported artifacts under `results/interpretable_model/{xgboost,gam}/.../tables`, primarily:

- `metrics_oof_ml_ade_log.csv`
- `nested_cv_optuna_summary_ml_ade_log.csv`
- `nested_variant_comparison_ml_ade_log.csv`
- `model_data_with_oof_ml_ade_log.csv`
- `run_manifest_ml_ade_log.json`

Although the artifact filenames reference `ml_ade_log`, the reported OOF and nested-CV metrics in these tables are on the original `ml_ade` scale. This matters for interpretation because the downstream performance-regime notebook clusters trajectories by raw `ml_ade` performance groups, while some exported feature-effect models are fitted in log-target mode.

## Downstream relevance

The feature effects from these meta-models are intended to explain, trajectory by trajectory, which trajectory characteristics drove the Trajectron++ `ml_ade` value. These local effects are then clustered within hard, medium, and easy performance groups to identify repeated explanation patterns.

For that use case, fit quality is not only a predictive-score question. A model that has weak OOF fit, poor rank preservation, or strong regression-to-the-mean will produce local effects that may cluster cleanly but explain the meta-model more than the underlying Trajectron++ behavior. The most relevant properties are therefore:

- OOF performance on held-out folds.
- Rank preservation across trajectories.
- Bias and compression across easy, medium, and hard regimes.
- Stability of the selected model across folds.
- Whether the model target mode matches the intended interpretation scale of the exported feature effects.

## Run setup

| Run | Model | Rows | Features | Selected model / target mode |
|---|---:|---:|---|---|
| `MI_correct` | XGBoost | 26,886 | `std_speed`, `max_speed`, `heading_change_per_sec`, `mean_acceleration` | XGBoost, log target |
| `MI_correct` | GAM | 26,886 | `std_speed`, `max_speed`, `heading_change_per_sec`, `mean_acceleration` | LinearGAM, log target |
| `vif_only_no_collision` | XGBoost | 26,122 | `std_speed`, `mean_acceleration`, `min_neighbor_distance`, `heading_change_per_sec`, `scene_num_VEHICLE`, `scene_density_VEHICLE` | XGBoost, log target |
| `vif_only_no_collision` | GAM | 26,122 | same as above | LinearGAM, raw target |

The row counts differ between the two runs, so differences between `MI_correct` and `vif_only_no_collision` should be read as the combined effect of feature selection and sample selection.

## Overall OOF fit

| Run | Model | OOF R2 | OOF MAE | OOF RMSE | Bias, pred - actual | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MI_correct` | XGBoost | 0.619 | 0.214 | 0.388 | -0.029 | 0.789 | 0.835 |
| `MI_correct` | GAM | 0.498 | 0.250 | 0.445 | -0.037 | 0.708 | 0.791 |
| `vif_only_no_collision` | XGBoost | 0.624 | 0.213 | 0.387 | -0.029 | 0.793 | 0.829 |
| `vif_only_no_collision` | GAM | 0.515 | 0.268 | 0.440 | 0.001 | 0.717 | 0.735 |

XGBoost is clearly the better-fitting meta-model in both runs. Its OOF R2 is about 0.62, with RMSE around 0.387-0.388 and Spearman rank correlation around 0.83. This is a reasonably strong surrogate fit for downstream local-effect analysis, especially because the model preserves trajectory ranking well.

The GAM fits are weaker. The `MI_correct` GAM reaches OOF R2 0.498 and Spearman 0.791. The `vif_only_no_collision` GAM improves R2 and RMSE slightly, but its MAE worsens and its Spearman correlation drops to 0.735. That means it captures some large-scale variation but is less reliable for trajectory-level ordering, which is important when clustering local explanations.

## Nested-CV stability

| Run | Model | Outer RMSE mean | Outer RMSE sd | Outer MAE mean | Outer MAE sd | Outer R2 mean | Outer R2 sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MI_correct` | XGBoost | 0.388 | 0.010 | 0.214 | 0.005 | 0.619 | 0.008 |
| `MI_correct` | GAM | 0.445 | 0.017 | 0.250 | 0.004 | 0.498 | 0.035 |
| `vif_only_no_collision` | XGBoost | 0.387 | 0.008 | 0.213 | 0.002 | 0.624 | 0.007 |
| `vif_only_no_collision` | GAM | 0.440 | 0.011 | 0.268 | 0.002 | 0.515 | 0.010 |

The XGBoost results are stable across folds in both runs. The `vif_only_no_collision` XGBoost fit is marginally more stable than `MI_correct`, but the difference is small.

The `MI_correct` GAM is noticeably less stable in R2 across folds. The `vif_only_no_collision` GAM is more stable and has better RMSE/R2, but this comes with worse MAE and weaker rank correlation.

## Performance-regime behavior

The performance-regime pipeline assigns easy, medium, and hard groups using k-means on `ml_ade` in log space. The resulting boundaries are almost identical across the two runs:

| Run | Easy/medium boundary | Medium/hard boundary | Easy n | Medium n | Hard n |
|---|---:|---:|---:|---:|---:|
| `MI_correct` | 0.388 | 1.198 | 15,823 | 8,320 | 2,743 |
| `vif_only_no_collision` | 0.388 | 1.200 | 15,348 | 8,094 | 2,680 |

Within-regime R2 is not a useful primary metric because the target range inside each group is narrow; it is frequently negative even when the global fit is acceptable. MAE, bias, and rank correlation are more informative.

### Raw-scale errors by performance group

| Run | Model | Group | Actual mean | Predicted mean | MAE | RMSE | Bias, pred - actual | Spearman |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `MI_correct` | XGBoost | easy | 0.177 | 0.258 | 0.109 | 0.177 | 0.081 | 0.698 |
| `MI_correct` | XGBoost | medium | 0.658 | 0.622 | 0.231 | 0.319 | -0.036 | 0.455 |
| `MI_correct` | XGBoost | hard | 2.028 | 1.383 | 0.773 | 0.993 | -0.645 | 0.480 |
| `MI_correct` | GAM | easy | 0.177 | 0.280 | 0.138 | 0.199 | 0.103 | 0.626 |
| `MI_correct` | GAM | medium | 0.658 | 0.611 | 0.241 | 0.360 | -0.048 | 0.383 |
| `MI_correct` | GAM | hard | 2.028 | 1.210 | 0.915 | 1.149 | -0.818 | 0.417 |
| `vif_only_no_collision` | XGBoost | easy | 0.177 | 0.261 | 0.112 | 0.179 | 0.084 | 0.685 |
| `vif_only_no_collision` | XGBoost | medium | 0.659 | 0.620 | 0.225 | 0.308 | -0.039 | 0.449 |
| `vif_only_no_collision` | XGBoost | hard | 2.034 | 1.389 | 0.761 | 0.996 | -0.645 | 0.483 |
| `vif_only_no_collision` | GAM | easy | 0.177 | 0.307 | 0.172 | 0.238 | 0.131 | 0.483 |
| `vif_only_no_collision` | GAM | medium | 0.659 | 0.657 | 0.255 | 0.357 | -0.002 | 0.376 |
| `vif_only_no_collision` | GAM | hard | 2.034 | 1.296 | 0.860 | 1.085 | -0.737 | 0.403 |

All models show regression-to-the-mean:

- Easy trajectories are over-predicted.
- Hard trajectories are strongly under-predicted.
- Medium trajectories are fit more closely in aggregate.

This is expected for a noisy performance target with a long right tail, but it matters for feature-effect clustering. The local effects will explain a compressed meta-model prediction, especially in the hard group. This compression is weaker for XGBoost and stronger for GAM.

## Individual assessments

### `full_trainval_12ep_1seed_MI_correct`

The `MI_correct` XGBoost model is a good fit for this downstream purpose. It explains about 62% of raw-scale OOF variance and preserves trajectory ranking well. It also has the best easy-group rank correlation among the compared fits and is consistently better than the corresponding GAM in every global metric.

The main weakness is hard-regime compression: hard trajectories average `ml_ade = 2.028`, but the XGBoost OOF predictions average only 1.383. This means the feature effects for hard trajectories likely understate the magnitude of the factors driving very poor Trajectron++ performance. However, because the model still has reasonable within-hard rank correlation, the effects remain usable for pattern discovery.

The `MI_correct` GAM is interpretable and reasonably fit, but substantially weaker than XGBoost. It has lower R2, higher RMSE, higher MAE, and lower rank preservation. It also under-predicts hard trajectories more strongly than XGBoost. The selected GAM variant is `gam-linear-log`; the nested comparison shows the log-linear and gamma GAM variants are almost tied on RMSE/R2, while the no-log linear variant is worse. The selected log-linear GAM is therefore defensible, but its feature effects should be treated as smoother, lower-resolution explanations.

### `full_trainval_12ep_1seed_vif_only_no_collision`

The `vif_only_no_collision` XGBoost model is the best overall fit in this comparison, but only marginally better than `MI_correct` XGBoost. Its OOF R2 increases from 0.619 to 0.624, RMSE decreases from 0.388 to 0.387, and MAE decreases from 0.214 to 0.213. These differences are small enough that they should not be over-interpreted, especially because this run has 764 fewer rows.

The feature set is more semantically useful for the intended explanation task because it includes neighborhood and scene-vehicle context (`min_neighbor_distance`, `scene_num_VEHICLE`, `scene_density_VEHICLE`) and drops `max_speed`. If the goal is to map clusters back to trajectory and scene characteristics, this is a practical advantage over the `MI_correct` feature set, provided the small sample change is acceptable.

The `vif_only_no_collision` GAM is mixed. It improves R2 and RMSE relative to `MI_correct` GAM, and it is more stable across folds. However, its MAE is worse and rank preservation drops materially. The selected variant is `gam-linear` with raw target mode, not log target mode. That target-mode change is important: the feature effects from this GAM are raw-scale ADE contributions, while the XGBoost feature effects and the `MI_correct` GAM are log-target effects. This makes direct effect-space comparisons across these GAM runs less clean.

## Direct comparison

### XGBoost: `MI_correct` vs. `vif_only_no_collision`

The two XGBoost fits are nearly tied. `vif_only_no_collision` is slightly better on raw-scale R2, RMSE, and MAE, while `MI_correct` has slightly better Spearman rank correlation. In practical terms, the difference in predictive fit is negligible.

For downstream clustering, `vif_only_no_collision` is preferable if the additional neighborhood and vehicle-scene features are substantively important for explaining Trajectron++ performance. It gives essentially the same fit quality as `MI_correct` while exposing more meaningful contextual features for cluster interpretation.

### GAM: `MI_correct` vs. `vif_only_no_collision`

The GAM comparison is less straightforward. `vif_only_no_collision` has better R2 and RMSE, but worse MAE and weaker rank preservation. It also changes the selected target mode from log to raw. Because the downstream work clusters local feature effects, rank preservation and effect-scale consistency are important. On those grounds, the `MI_correct` GAM is cleaner for log-effect interpretation, while the `vif_only_no_collision` GAM is better only if raw-scale GAM effects are specifically desired.

### XGBoost vs. GAM

XGBoost dominates GAM in both run configurations. The advantage is large enough to affect the credibility of downstream explanation clusters:

- In `MI_correct`, XGBoost improves OOF R2 by about 0.121 and reduces RMSE by about 0.057.
- In `vif_only_no_collision`, XGBoost improves OOF R2 by about 0.109 and reduces RMSE by about 0.053.
- XGBoost has much stronger rank preservation, especially for `vif_only_no_collision`.
- XGBoost compresses hard trajectories less severely than GAM.

The tradeoff is interpretability. GAM effects are smoother and easier to inspect as additive curves, but the lower fit means their local effects are less faithful to the actual variation in Trajectron++ performance.

## Recommendation

For the feature-effect performance-regime clustering, the strongest candidate is:

**XGBoost with `full_trainval_12ep_1seed_vif_only_no_collision`.**

The reason is not that it massively outperforms `MI_correct` XGBoost. It does not. The reason is that it preserves essentially the same strong OOF fit while using a feature set that is more useful for explaining pedestrian trajectory prediction performance: speed/acceleration, turning behavior, neighbor distance, and scene vehicle context.

The main caution is hard-regime compression. Even the best XGBoost fit under-predicts hard trajectories by about 0.645 ADE on average. Clusters in the hard regime should therefore be interpreted as patterns in the model's explanation of poor performance, not as a perfect decomposition of the full observed ADE magnitude.

The GAM fits can still be useful as a sensitivity check or for producing smoother explanatory summaries. However, they should not be the primary source for clustering if the goal is trajectory-level explanation fidelity. If GAM effects are used, avoid mixing log-target and raw-target GAM effects without explicitly normalizing the interpretation, because the selected GAM target mode differs between the two runs.

