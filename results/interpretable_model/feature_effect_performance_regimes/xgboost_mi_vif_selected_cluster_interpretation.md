# XGBoost MI+VIF selected cluster interpretation

Date: 2026-05-17

## Scope

This report interprets the selected/best clustering for each performance group in the XGBoost run trained on the MI+VIF feature set:

`xgboost/full_trainval_12ep_1seed_MI_correct/ml_ade_log`

The selected clusterings are the `selected_for_group` rows exported by `feature_effect_performance_regimes.ipynb`. The clustering space is the signed feature-effect vector for these four features:

- `std_speed`
- `max_speed`
- `heading_change_per_sec`
- `mean_acceleration`

Positive feature effects increase the XGBoost meta-model prediction of `ml_ade_log`, so they indicate a contribution toward worse Trajectron++ prediction performance. Negative feature effects decrease the prediction.

Raw trajectory values are reported as medians. Offsets such as `+2.42 global IQR` mean that the cluster median is 2.42 interquartile ranges above the global median across all performance groups.

## How To Read This Report

Read each performance group in three layers. First, inspect the selected clustering table to understand whether the clustering itself is strong: high DBCV, moderate noise, and non-tiny clusters make a cluster interpretation more reliable. Second, read the per-cluster table and prose to see whether feature effects and raw trajectory characteristics tell the same story. Third, use the whole-group interpretation when the selected clusters are weak, tiny, or mostly noise.

Key figures:

- `ml_ade` median: the median Trajectron++ average displacement error in the component. Lower is better. Within a performance group, a higher cluster median means that component is harder even relative to its group.
- Mean feature effects: the average signed contribution of each trajectory feature to the XGBoost meta-model prediction of `ml_ade_log`. Positive values push predicted error up; negative values push predicted error down.
- Median total feature effect: the median row-wise sum of the signed feature effects across the clustering features. A negative value means the feature effects usually reduce predicted error; a positive value means they usually increase predicted error. For example, the easy group has -0.195, while the hard group has +0.525.
- Positive total-effect share: the share of rows whose summed feature effects are positive. This tells whether the group is consistently pushed toward higher predicted error or only weakly/mixed. For example, 99.5% in hard means almost all hard rows have feature effects that increase predicted error; 6.4% in easy means almost all easy rows have feature effects that lower predicted error.
- Raw trajectory medians: the actual trajectory-characteristic values behind the feature effects. These are essential because a feature-effect cluster is only semantically useful if the underlying trajectories also form an interpretable pattern.
- Global IQR offsets: robust relative positions of raw values versus all pedestrian trajectories in the run. `+1.00 global IQR` means the cluster median is one interquartile range above the global median; `-1.00` means one IQR below.
- Noise: points not assigned to a non-noise cluster by OPTICS. Noise is not treated as a validated cluster, but if it has coherent feature effects and raw characteristics it can still describe a broad continuum or boundary region.
- Confidence: high means the pattern is large, coherent, and supported by the selected XGBoost MI+VIF clustering. Medium means it is better seen as a group-level pattern or is supported by complementary runs. Low means it is tiny, noisy, or not stable enough for a main stakeholder narrative.

## Selected Clusterings

| group | selected candidate | clusters | noise | DBCV | largest cluster |
| --- | --- | --- | --- | --- | --- |
| easy | `cluster_optics_raw__mcs-frac-0p0015__mcs-24__ms-40__eps-0p01808` | 2 | 0.300 | 0.411 | 0.364 |
| medium | `cluster_optics_raw__mcs-frac-0p0015__mcs-13__ms-15__eps-0p02425` | 2 | 0.400 | 0.397 | 0.598 |
| hard | `cluster_optics_raw__mcs-frac-0p0015__mcs-10__ms-15__eps-0p04134` | 4 | 0.748 | 0.091 | 0.172 |

The easy clustering is the strongest selected clustering. The medium clustering has good DBCV but weak semantic value because one cluster is tiny and the higher-error medium cases sit mostly in noise. The hard clustering has weak cluster quality; the hard group and hard noise provide a stronger explanation than the named clusters.

## Easy Group

The easy selected clustering has two large non-noise clusters plus 30.0% noise. Both clusters are interpretable, but they represent different reasons why trajectories remain easy.

| component | n / share | `ml_ade` median | median total effect | mean feature effects | raw trajectory medians | concise interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| whole group | 15,823 / 100% | 0.167 | -0.195 | `std_speed` -0.083, `max_speed` -0.061, `heading_change_per_sec` -0.041, `mean_acceleration` +0.003 | `std_speed` 0.0677 (-0.28 global IQR), `max_speed` 1.190 (-0.15), `heading_change_per_sec` 5.02 (-0.04), `mean_acceleration` -0.0006 | Broadly low-error; feature effects mostly reduce predicted error. |
| cluster 0 | 5,323 / 33.6% | 0.083 | -0.282 | `std_speed` -0.128, `max_speed` -0.223, `heading_change_per_sec` +0.041, `mean_acceleration` +0.023 | `std_speed` 0.0366 (-0.55 global IQR), `max_speed` 0.124 (-1.12), `heading_change_per_sec` 45.7 (+1.42), `mean_acceleration` 0.000 | Slow, high-turning trajectories that remain easy because low speed dominates. |
| cluster 1 | 5,754 / 36.4% | 0.199 | -0.156 | `std_speed` -0.081, `max_speed` +0.042, `heading_change_per_sec` -0.121, `mean_acceleration` -0.000 | `std_speed` 0.0661 (-0.30 global IQR), `max_speed` 1.412 (+0.05), `heading_change_per_sec` 1.93 (-0.16), `mean_acceleration` -0.0027 | Mostly straight, stable-heading trajectories. |
| noise | 4,746 / 30.0% | 0.238 | -0.090 | `std_speed` -0.035, `max_speed` -0.005, `heading_change_per_sec` -0.036, `mean_acceleration` -0.014 | `std_speed` 0.128 (+0.23 global IQR, +0.78 within easy), `max_speed` 1.349 (-0.01), `heading_change_per_sec` 7.05 (+0.03), `mean_acceleration` -0.0022 | Heterogeneous easy-boundary cases, more dynamic than the named clusters. |

### Easy Per-Cluster Interpretation

Cluster 0 is the clearest easy subgroup. It contains 33.6% of the easy group and has very low prediction error (`ml_ade` 0.083 vs easy median 0.167). The net feature effect is strongly negative (-0.282), driven by very negative `max_speed` (-0.223) and `std_speed` (-0.128) effects. The raw trajectory values explain why: `max_speed` is only 0.124 (-1.12 global IQR), `std_speed` is 0.0366 (-0.55 global IQR), and displacement/path length are also low. The apparent complication is high turning (`heading_change_per_sec` 45.7, +1.42 global IQR), which contributes positively to error (+0.041), but the pedestrian is moving so slowly that low speed dominates.

Cluster 1 is also coherent, but it is a different kind of easy case. It contains 36.4% of easy rows and has `ml_ade` 0.199, still within the easy group but higher than cluster 0. The total feature effect is negative (-0.156), mainly because `heading_change_per_sec` has a strong negative effect (-0.121). The raw trajectories have very low turning (`heading_change_per_sec` 1.93, -0.16 global IQR), normal maximum speed (`max_speed` 1.412), and low speed variation (`std_speed` 0.0661, -0.30 global IQR).

Easy noise is not a clean trajectory type. It is the hardest part of the easy group (`ml_ade` 0.238) and has less negative total effect (-0.090). It has higher speed variation than the named easy clusters (`std_speed` 0.128, +0.78 within easy), and acceleration/jerk are elevated relative to easy. It is best interpreted as heterogeneous easy-boundary traffic: still easy overall, but too dynamically mixed for a stable cluster story.

### Easy Whole-Group Interpretation

The whole easy group is a robust low-error regime, as shown by the negative whole-group total effect in the table. The group-level view is useful, but the selected clustering adds clearer stakeholder narratives than the average because it separates two different easy mechanisms: "slow despite turning" and "straight with stable heading."

Complementary runs do not change this story. GAM/MI+VIF and XGBoost HDBSCAN alternatives mostly split the same easy structure into slow or stationary subclusters.

## Medium Group

The medium selected clustering is numerically strong by DBCV, but semantically weaker. The main non-noise cluster is a broad typical-medium cluster; the second cluster is tiny; the higher-error medium cases are mostly noise.

| component | n / share | `ml_ade` median | median total effect | mean feature effects | raw trajectory medians | concise interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| whole group | 8,320 / 100% | 0.602 | +0.057 | `std_speed` +0.024, `max_speed` +0.064, `heading_change_per_sec` -0.002, `mean_acceleration` -0.003 | `std_speed` 0.145 (+0.38 global IQR), `max_speed` 1.515 (+0.14), `heading_change_per_sec` 5.75 (-0.02), `mean_acceleration` -0.0006 | Broad transition regime; mildly positive speed-related difficulty. |
| cluster 0 | 4,975 / 59.8% | 0.551 | +0.008 | `std_speed` -0.033, `max_speed` +0.072, `heading_change_per_sec` -0.028, `mean_acceleration` +0.007 | `std_speed` 0.122 (+0.18 global IQR), `max_speed` 1.525 (+0.15), `heading_change_per_sec` 4.64 (-0.06), `mean_acceleration` -0.0018 | Ordinary medium cases; max speed adds mild error pressure. |
| cluster 1 | 19 / 0.2% | 0.424 | -0.207 | `std_speed` -0.125, `max_speed` -0.226, `heading_change_per_sec` +0.107, `mean_acceleration` +0.037 | `std_speed` 0.0669 (-0.29 global IQR), `max_speed` 0.190 (-1.06), `heading_change_per_sec` 48.6 (+1.52), `mean_acceleration` 0.0049 | Tiny slow/high-turning edge case, not a robust medium segment. |
| noise | 3,326 / 40.0% | 0.726 | +0.187 | `std_speed` +0.109, `max_speed` +0.054, `heading_change_per_sec` +0.036, `mean_acceleration` -0.017 | `std_speed` 0.224 (+1.06 global IQR, +0.73 within medium), `max_speed` 1.491 (+0.12), `heading_change_per_sec` 10.5 (+0.15 global, +0.65 within medium), `mean_acceleration` 0.0035 | More dynamic higher-error medium continuum; not a stable cluster. |

### Medium Per-Cluster Interpretation

Cluster 0 is the main medium cluster by size, but it is not a crisp trajectory type. It contains 59.8% of the medium group, has `ml_ade` below the medium median (0.551 vs 0.602), and has almost neutral total feature effect (+0.008). The only clearly positive mean effect is `max_speed` (+0.072; median +0.070), and 99% of rows have positive `max_speed` effects. The raw values are close to typical medium trajectories: `max_speed` is 1.525, only +0.15 global IQR, `std_speed` is 0.122, only +0.18 global IQR, and `heading_change_per_sec` is 4.64, slightly below the global median. This cluster is best read as ordinary medium cases where maximum speed adds some error pressure, but stable heading and only moderate speed variation keep the prediction quality from becoming hard.

Cluster 1 is coherent but too small to communicate as a robust stakeholder segment. It has only 19 rows, low `ml_ade` for the medium group (0.424), and negative total effect (-0.207). The raw profile is extreme: very low `max_speed` (0.190, -1.06 global IQR), low `std_speed` (0.0669), but very high `heading_change_per_sec` (48.6, +1.52 global IQR and +5.82 within medium). The interpretation is "slow pedestrians with many turns"; turning increases predicted error, but the low speed and low speed variation dominate, making these rows easier than typical medium rows. Because the cluster is tiny, it should be treated as an edge case, not a medium-regime pattern.

The noise portion is semantically more important than either named cluster. It contains 40.0% of medium rows, has the highest medium `ml_ade` median (0.726), and has a clearly positive total feature effect (+0.187). Its raw `std_speed` is 0.224 (+1.06 global IQR, +0.73 within medium), `heading_change_per_sec` is 10.5 (+0.65 within medium), and acceleration/jerk are also elevated (`max_acceleration` is +0.72 global IQR; `mean_jerk` and `max_jerk` are each about +0.62 global IQR). This is the useful medium story: the more difficult medium trajectories are more dynamic, with changing speeds and more turning, but they form a continuum rather than a stable cluster.

### Medium Whole-Group Interpretation

The whole medium group provides the best communicable medium-regime picture. The table shows a mildly positive total effect and only moderate raw shifts: `std_speed` is above the global median, `max_speed` is slightly above, and `heading_change_per_sec` is essentially typical. This is a transition zone where some features push error up but not enough to produce consistently hard predictions. Acceleration and jerk are also moderately elevated outside the clustering features (`max_acceleration` +0.33 global IQR, `mean_jerk` +0.32, `max_jerk` +0.27), but `mean_acceleration` effects are small.

**Medium interpretation:** the selected clusters do not produce a strong multi-cluster story. Medium trajectories are best communicated as a broad transition regime: maximum speed and some speed variability make prediction harder than easy cases, but heading changes are not consistently high and the total effect remains only mildly positive. The higher-error medium cases are the dynamic/noisy part of this continuum, not a stable exported cluster.

Complementary runs agree. GAM/MI+VIF medium is dominated by one large cluster and tiny low-speed clusters; GAM/VIF-only medium is also weak. XGBoost/VIF-only has no exported medium clustering, but its whole-group profile also points to modest positive `std_speed` and heading-change effects. There is no strong alternative medium narrative.

## Hard Group

The selected hard clustering should be interpreted cautiously. It has four non-noise clusters, but 74.8% of hard rows are noise and DBCV is only 0.091. The non-noise clusters are small and mostly lower-error than the hard-group median. The main hard-regime story is carried by the hard group as a whole and by the noise rows.

| component | n / share | `ml_ade` median | median total effect | mean feature effects | raw trajectory medians | concise interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| whole group | 2,743 / 100% | 1.782 | +0.525 | `std_speed` +0.296, `max_speed` +0.127, `heading_change_per_sec` +0.128, `mean_acceleration` +0.001 | `std_speed` 0.333 (+2.00 global IQR), `max_speed` 1.525 (+0.15), `heading_change_per_sec` 14.9 (+0.31), `mean_acceleration` 0.0023 (+0.09) | Broadly high-error; effects almost always increase predicted error. |
| cluster 0 | 31 / 1.1% | 1.942 | +0.675 | `std_speed` +0.488, `max_speed` +0.070, `heading_change_per_sec` +0.126, `mean_acceleration` -0.019 | `std_speed` 0.452 (+3.02 global IQR), `max_speed` 1.244 (-0.10), `heading_change_per_sec` 12.3 (+0.22), `mean_acceleration` -0.112 (-3.60) | Tiny high-speed-variation subgroup; partly coherent. |
| cluster 1 | 68 / 2.5% | 1.584 | +0.507 | `std_speed` +0.381, `max_speed` +0.075, `heading_change_per_sec` +0.149, `mean_acceleration` -0.077 | `std_speed` 0.389 (+2.48), `max_speed` 1.194 (-0.15), `heading_change_per_sec` 27.2 (+0.75), `mean_acceleration` -0.121 (-3.89) | Irregular motion, but below hard-median error. |
| cluster 2 | 118 / 4.3% | 1.607 | +0.450 | `std_speed` +0.233, `max_speed` +0.066, `heading_change_per_sec` +0.130, `mean_acceleration` +0.012 | `std_speed` 0.288 (+1.61), `max_speed` 1.074 (-0.26), `heading_change_per_sec` 19.5 (+0.48), `mean_acceleration` -0.0287 (-0.91) | Milder high-variation/high-turning subgroup. |
| cluster 3 | 473 / 17.2% | 1.500 | +0.325 | `std_speed` +0.009, `max_speed` +0.136, `heading_change_per_sec` +0.146, `mean_acceleration` +0.031 | `std_speed` 0.127 (+0.23), `max_speed` 1.551 (+0.17), `heading_change_per_sec` 10.5 (+0.15), `mean_acceleration` -0.0037 (-0.10) | Weak semantic cluster; lower-error hard-boundary cases. |
| noise | 2,053 / 74.8% | 1.934 | +0.601 | `std_speed` +0.361, `max_speed` +0.132, `heading_change_per_sec` +0.123, `mean_acceleration` -0.004 | `std_speed` 0.382 (+2.42 global IQR), `max_speed` 1.539 (+0.16), `heading_change_per_sec` 18.8 (+0.45), `mean_acceleration` 0.017 (+0.58); `max_acceleration` is also high (+1.15 global IQR) | Main hard-regime story: dynamic, high-error continuum. |

### Hard Per-Cluster Interpretation

Cluster 0 is a tiny but partly coherent high-variation subgroup. It contains only 31 rows (1.1% of hard) and has `ml_ade` 1.942, above the hard median of 1.782. The feature effects are strongly positive overall (+0.675), especially `std_speed` (+0.488), with additional positive `heading_change_per_sec` (+0.126) and `max_speed` (+0.070). The raw profile is extreme in speed variation (`std_speed` 0.452, +3.02 global IQR), but not in maximum speed (`max_speed` 1.244, -0.10 global IQR) or turning (`heading_change_per_sec` 12.3, +0.22 global IQR). `mean_acceleration` is strongly negative (-3.60 global IQR). This is a plausible extreme-speed-variation hard case, but it is too small to be a main segment.

Cluster 1 is coherent as a high-turning/high-variation motion type, but it does not map to higher hard-regime error. It has 68 rows (2.5%), positive total feature effect (+0.507), high `std_speed` (+2.48 global IQR), high heading change (`heading_change_per_sec` 27.2, +0.75 global IQR), and strongly negative mean acceleration (-3.89 global IQR). The issue is that `ml_ade` is 1.584, below the hard median. This means it is semantically interpretable as irregular motion, but not as a strong explanation of the worst hard cases.

Cluster 2 is a milder version of cluster 1. It has 118 rows (4.3%), positive effects for `std_speed` (+0.233), `heading_change_per_sec` (+0.130), and `max_speed` (+0.066), and raw speed variation is high (`std_speed` 0.288, +1.61 global IQR). It also has below-hard-median `ml_ade` (1.607). It can be described as moderate speed-variation/turning difficulty, but it is not a strong stakeholder segment.

Cluster 3 is the largest non-noise hard cluster (17.2%), but it is the weakest semantic cluster. Its feature effects are positive for `max_speed` (+0.136) and `heading_change_per_sec` (+0.146), yet the raw values are only mildly shifted: `std_speed` 0.127 (+0.23 global IQR), `max_speed` 1.551 (+0.17), and `heading_change_per_sec` 10.5 (+0.15). Its `ml_ade` is 1.500, well below the hard median. This should not be communicated as a hard trajectory type; it is a lower-error hard-boundary cluster.

Hard noise is the main hard explanation. It contains 74.8% of hard rows, has higher `ml_ade` than the group median (1.934 vs 1.782), and a strongly positive total effect (+0.601). The raw values are aligned with the feature effects: `std_speed` is 0.382 (+2.42 global IQR), `heading_change_per_sec` is 18.8 (+0.45 global IQR), `mean_acceleration` is +0.58 global IQR, and `max_acceleration` is +1.15 global IQR.

### Hard Whole-Group Interpretation

The whole hard group is more communicable than the selected non-noise clusters. The table shows that hard is broadly characterized by positive error contributions, especially from `std_speed`, `heading_change_per_sec`, and `max_speed`. The hard/noise raw values show the corresponding trajectory characteristics: high speed variation, more turning, and elevated acceleration/jerk. The named clusters can be used as examples of small edge cases, but the whole-group/noise explanation is the defensible stakeholder story.

Complementary runs add value for hard. GAM/MI+VIF has a small hard cluster with high `std_speed`, high heading change, high acceleration, and high `ml_ade` (`ml_ade` median 2.457). XGBoost/VIF-only and GAM/VIF-only also isolate tiny extreme-hard clusters with very high speed variation and high `ml_ade` (`3.384` and `2.442`, respectively). These support the existence of extreme hard outliers, but they do not turn the XGBoost MI+VIF hard clustering into a stable multi-cluster segmentation.

## VIF-Only Added-Feature Check

The VIF-only runs add `min_neighbor_distance`, `scene_num_VEHICLE`, and for XGBoost also `scene_density_VEHICLE` to the clustering feature set. I checked whether those features add explanatory value beyond the MI+VIF trajectory features.

They do not produce a robust additional stakeholder narrative. In the XGBoost VIF-only hard selected clustering, `min_neighbor_distance` appears as the third-largest effect in the broad lower-error hard cluster, but the effect is small: +0.047, only about 17% of the dominant `std_speed` effect. The underlying raw `min_neighbor_distance` is almost unchanged (-0.03 within-group IQR and +0.09 global IQR), and the cluster is lower-error than the hard median (`ml_ade` 1.571 vs 1.783). That means the model assigns a small positive neighbor-distance effect, but the trajectories in that cluster are not meaningfully different in neighbor distance.

The vehicle features are even weaker. Across exported VIF-only candidates, `scene_num_VEHICLE` appears only as a small third effect in tiny alternatives: mostly GAM/VIF-only hard alternatives of about 1-4% of hard rows and about 5% of the dominant effect size, plus a few extremely tiny XGBoost/VIF-only alternatives where it is about 0.2% of hard rows and about 1.5% of the dominant effect size. The raw vehicle-count shifts are small. `scene_density_VEHICLE` is not semantically relevant in exported clusters; for GAM it was not used for clustering because it was not significant, and in XGBoost it only appears in tiny lower-quality hard alternatives.

So the added VIF-only features are at most weak secondary descriptors. They do not explain why Trajectron++ performs well or poorly as clearly as `std_speed`, `max_speed`, `heading_change_per_sec`, and acceleration/jerk. The complementary value of the VIF-only runs is instead that they corroborate tiny extreme-hard speed-variation outliers.

## Stakeholder Summary

| performance group | basis | communicable trajectory type | explanation pattern | confidence |
| --- | --- | --- | --- | --- |
| easy | XGBoost MI+VIF selected easy cluster 0 (`n=5,323`, 33.6% of easy). Supported by similar slow/stationary splits in GAM/MI+VIF and HDBSCAN alternatives. | Slow pedestrians who turn a lot but barely move fast. `max_speed` is only 0.124 (-1.12 global IQR), `std_speed` is 0.0366 (-0.55), and `heading_change_per_sec` is high at 45.7 (+1.42). | Turning alone would make prediction harder (`heading_change_per_sec` effect +0.041), but very low speed and low speed variation strongly reduce predicted error (`max_speed` -0.223, `std_speed` -0.128). Net effect is strongly easy (`ml_ade` 0.083; total effect -0.282). | High. This is a large, coherent selected XGBoost MI+VIF cluster and is supported by alternative easy clusterings. |
| easy | XGBoost MI+VIF selected easy cluster 1 (`n=5,754`, 36.4% of easy). | Mostly straight pedestrians with normal maximum speed. `heading_change_per_sec` is 1.93 (-0.16 global IQR), `max_speed` is 1.412, and `std_speed` is 0.0661. | Stable heading is the main reason these trajectories are easy (`heading_change_per_sec` effect -0.121). Maximum speed contributes slightly toward error (+0.042), but not enough to overcome the low-turning benefit. `ml_ade` remains low at 0.199. | High. Large selected cluster with coherent effects and raw characteristics. |
| medium | XGBoost MI+VIF whole medium group (`n=8,320`) plus selected medium cluster 0 as the ordinary-medium component (`n=4,975`, 59.8%). | Typical medium trajectories with normal-to-slightly-high speed but no extreme turning. Whole group values: `ml_ade` 0.602, `std_speed` 0.145 (+0.38 global IQR), `max_speed` 1.515 (+0.14), and `heading_change_per_sec` 5.75 (near global median). | Medium is a transition regime. `max_speed` is the most consistent positive effect (+0.066 median; 90.3% positive), while heading and speed-variation effects are mixed. The total effect is only mildly positive (+0.057). | Medium. This is better supported by the whole group than by discrete clusters. |
| medium | XGBoost MI+VIF selected medium noise (`n=3,326`, 40.0% of medium). This is not a stable cluster but is the clearest higher-error medium component. | More dynamic medium-boundary trajectories. `ml_ade` is higher at 0.726, `std_speed` is 0.224 (+1.06 global IQR), `heading_change_per_sec` is 10.5, and acceleration/jerk are elevated. | Changing speeds and more turning push trajectories toward worse prediction quality (`std_speed` +0.109, `heading_change_per_sec` +0.036, total effect +0.187), but the structure is continuous and not a stable cluster. | Medium-low as a cluster, medium as a regime-level explanation. It is noise, but it is coherent and aligns with the whole-group transition story. |
| hard | XGBoost MI+VIF whole hard group (`n=2,743`) and selected hard noise (`n=2,053`, 74.8% of hard). | Broad high-variation hard trajectories. For the hard group as a whole, `ml_ade` is 1.782, median total effect is +0.525, and 99.5% of rows have positive total effects. Hard noise is especially clear: `std_speed` 0.382 (+2.42 global IQR), `heading_change_per_sec` 18.8 (+0.45), and `max_acceleration` +1.15 global IQR. | Trajectron++ struggles when pedestrian motion varies strongly in speed and direction. XGBoost assigns positive effects to `std_speed` (+0.361 in hard noise), `max_speed` (+0.132), and `heading_change_per_sec` (+0.123), yielding higher `ml_ade` (1.934 in hard noise). | High as a whole-group/noise explanation, low as a discrete-cluster explanation. |
| hard | XGBoost MI+VIF selected hard cluster 0 (`n=31`, 1.1% of hard), complemented by XGBoost/VIF-only hard cluster 1 (`ml_ade` 3.384) and GAM/VIF-only hard cluster 1 (`ml_ade` 2.442). | Tiny extreme-hard outliers with very high speed variation and often high acceleration/turning. XGBoost MI+VIF hard cluster 0 has `std_speed` 0.452 (+3.02 global IQR) and `ml_ade` 1.942. VIF-only runs isolate even more extreme tiny clusters. | These are the rare cases where motion is so irregular that all runs tend to flag them as difficult. The explanation pattern is extreme speed variation, often combined with turning and acceleration/jerk. | Medium. The pattern is consistent across runs, but the clusters are tiny and should be described as outliers, not a large hard subtype. |
