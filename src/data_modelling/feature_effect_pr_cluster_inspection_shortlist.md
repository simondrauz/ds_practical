# Feature-Effect Cluster Inspection Shortlist

This file lists the empirical-sweep clusterings selected for semantic inspection in
`feature_effect_pr_cluster_inspection.ipynb`.

Use the `CLUSTER_SPEC_DIRNAME` value from the relevant model section to avoid
automatic cluster-spec resolution. Then set:

```python
INSPECTION_CONFIG["performance_group"] = "<performance_group>"
INSPECTION_CONFIG["inspection_algorithm"] = "<algorithm>"
INSPECTION_CONFIG["inspection_cluster_space"] = "raw"
INSPECTION_CONFIG["cluster_ids"] = "all"
```

The `candidate_label_col` is included because runner-up clusterings can share the
same `(performance_group, algorithm, cluster_space)` as the tier-1 candidate.
The current inspection notebook resolves the selected candidate automatically
from `cluster_scores.csv`; rows marked `direct_config_selectable = yes` are
selectable with the current config alone. Rows marked `no` need a candidate-label
override or equivalent manual selection before the exact runner-up can be
inspected.

Selection basis:

- Source tables: downstream `tables/cluster_scores.csv` from the empirical sweep exports.
- Tier 1: `selected_for_group == True`.
- Runner-up 1: next best distinct exported partition by the same quality-first
  ranking used for group selection.
- Runner-up 2: best distinct alternate-algorithm partition when available;
  otherwise the next best distinct exported partition.
- Structure alternates: additional distinct exported partitions with a similar
  quality range or a materially different cluster-size/noise tradeoff worth
  semantic inspection.
- Ranking keys: valid selection, quality issue count, descending raw-effect DBCV,
  noise fraction, cluster count, then deterministic algorithm/settings tie-breaks.
- Duplicate hyperparameter rows that produce the same cluster partition are not
  repeated.

## GAM Additive Effects

```python
MODEL_ID = "gam"
RUN_NAME = "full_trainval_12ep_1seed"
CLUSTER_SPEC_DIRNAME = "cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf"
```

Cluster-spec root:
`results/interpretable_model/feature_effect_performance_regimes/gam/full_trainval_12ep_1seed/ml_ade_log/target-ml_ade_log__eval-eval_epoch_12.csv__lower-is-better-true__group-col-performance_group/cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf`

| performance_group | tier | algorithm | direct_config_selectable | n_clusters | noise_fraction | largest_cluster_share | dbcv_raw_effect_space | non_noise_cluster_sizes | setting | candidate_label_col | CLUSTER_SPEC_DIRNAME |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| easy | tier_1 | optics | yes | 3 | 0.114 | 0.501 | 0.553 | 7706/5269/642 | xi=0.005, mcs_frac=0.04, mcs=616, ms=10 | `cluster_optics_raw__mcs-frac-0p04__mcs-616__ms-10__xi-0p005` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| easy | runner_up_1 | optics | no | 3 | 0.250 | 0.410 | 0.543 | 6312/4598/625 | eps=0.032064, mcs_frac=0.003, mcs=47, ms=40 | `cluster_optics_raw__mcs-frac-0p003__mcs-47__ms-40__eps-0p03206` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| easy | runner_up_2 | hdbscan | yes | 4 | 0.081 | 0.536 | 0.334 | 8236/5334/356/200 | mcs_frac=0.01, mcs=154, ms=3 | `cluster_hdbscan_raw__mcs-frac-0p01__mcs-154__ms-3` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| easy | structure_alt_1 | optics | no | 3 | 0.410 | 0.486 | 0.440 | 7480/949/642 | xi=0.005, mcs_frac=0.04, mcs=616, ms=15 | `cluster_optics_raw__mcs-frac-0p04__mcs-616__ms-15__xi-0p005` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| medium | tier_1 | optics | yes | 2 | 0.256 | 0.703 | 0.238 | 5707/329 | xi=0.005, mcs_frac=0.04, mcs=325, ms=20 | `cluster_optics_raw__mcs-frac-0p04__mcs-325__ms-20__xi-0p005` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| medium | runner_up_1 | optics | no | 2 | 0.200 | 0.781 | 0.197 | 6341/156 | eps=0.068086, mcs_frac=0.003, mcs=25, ms=40 | `cluster_optics_raw__mcs-frac-0p003__mcs-25__ms-40__eps-0p06809` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| medium | runner_up_2 | hdbscan | yes | 2 | 0.220 | 0.768 | 0.051 | 6231/103 | mcs_frac=0.003, mcs=25, ms=40 | `cluster_hdbscan_raw__mcs-frac-0p003__mcs-25__ms-40` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| hard | tier_1 | optics | yes | 2 | 0.197 | 0.785 | 0.439 | 2110/48 | eps=0.13874, mcs_frac=0.003, mcs=20, ms=40 | `cluster_optics_raw__mcs-frac-0p003__mcs-20__ms-40__eps-0p1387` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| hard | runner_up_1 | optics | no | 2 | 0.198 | 0.781 | 0.437 | 2099/55 | eps=0.12777, mcs_frac=0.003, mcs=20, ms=30 | `cluster_optics_raw__mcs-frac-0p003__mcs-20__ms-30__eps-0p1278` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| hard | runner_up_2 | hdbscan | yes | 2 | 0.197 | 0.783 | 0.247 | 2103/55 | mcs_frac=0.003, mcs=20, ms=40 | `cluster_hdbscan_raw__mcs-frac-0p003__mcs-20__ms-40` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |
| hard | structure_alt_1 | optics | no | 2 | 0.161 | 0.812 | 0.364 | 2182/72 | xi=0.02, mcs_frac=0.02, mcs=54, ms=20 | `cluster_optics_raw__mcs-frac-0p02__mcs-54__ms-20__xi-0p02` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs8-c5f32c-floor20-ms8-fabe32__18d930323faf` |

## XGBoost SHAP Effects

```python
MODEL_ID = "xgboost"
RUN_NAME = "full_trainval_12ep_1seed"
CLUSTER_SPEC_DIRNAME = "cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5"
```

Cluster-spec root:
`results/interpretable_model/feature_effect_performance_regimes/xgboost/full_trainval_12ep_1seed/ml_ade_log/target-ml_ade_log__eval-eval_epoch_12.csv__lower-is-better-true__group-col-performance_group/cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5`

| performance_group | tier | algorithm | direct_config_selectable | n_clusters | noise_fraction | largest_cluster_share | dbcv_raw_effect_space | non_noise_cluster_sizes | setting | candidate_label_col | CLUSTER_SPEC_DIRNAME |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| easy | tier_1 | optics | yes | 2 | 0.300 | 0.355 | 0.272 | 5454/5312 | eps=0.024156, mcs_frac=0.0015, mcs=24, ms=30 | `cluster_optics_raw__mcs-frac-0p0015__mcs-24__ms-30__eps-0p02416` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| easy | runner_up_1 | hdbscan | yes | 11 | 0.389 | 0.455 | 0.158 | 6991/1661/138/137/109/77/74/68/63/51/31 | mcs_frac=0.002, mcs=31, ms=8 | `cluster_hdbscan_raw__mcs-frac-0p002__mcs-31__ms-8` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| easy | runner_up_2 | hdbscan | no | 8 | 0.378 | 0.455 | 0.143 | 6991/1661/471/138/110/74/63/51 | mcs_frac=0.003, mcs=47, ms=8 | `cluster_hdbscan_raw__mcs-frac-0p003__mcs-47__ms-8` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| easy | structure_alt_1 | hdbscan | no | 7 | 0.281 | 0.453 | 0.143 | 6964/3792/73/68/59/56/51 | mcs_frac=0.0015, mcs=24, ms=30 | `cluster_hdbscan_raw__mcs-frac-0p0015__mcs-24__ms-30` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| easy | structure_alt_2 | optics | no | 2 | 0.200 | 0.429 | 0.135 | 6592/5713 | eps=0.024925, mcs_frac=0.0015, mcs=24, ms=15 | `cluster_optics_raw__mcs-frac-0p0015__mcs-24__ms-15__eps-0p02492` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| medium | tier_1 | optics | yes | 3 | 0.350 | 0.647 | 0.336 | 5249/25/4 | eps=0.036559, mcs_frac=0.0015, mcs=13, ms=15 | `cluster_optics_raw__mcs-frac-0p0015__mcs-13__ms-15__eps-0p03656` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| medium | runner_up_1 | hdbscan | yes | 2 | 0.282 | 0.714 | 0.273 | 5796/28 | mcs_frac=0.0015, mcs=13, ms=30 | `cluster_hdbscan_raw__mcs-frac-0p0015__mcs-13__ms-30` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| medium | runner_up_2 | hdbscan | no | 3 | 0.353 | 0.638 | 0.059 | 5175/54/24 | mcs_frac=0.002, mcs=17, ms=10 | `cluster_hdbscan_raw__mcs-frac-0p002__mcs-17__ms-10` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |
| hard | tier_1 | optics | yes | 4 | 0.748 | 0.179 | 0.035 | 481/132/34/29 | eps=0.064755, mcs_frac=0.0015, mcs=10, ms=30 | `cluster_optics_raw__mcs-frac-0p0015__mcs-10__ms-30__eps-0p06476` | `cluster_spec__groups-easy-medium-hard__algs-hdbscan-optics__spaces-raw__metrics-fc851a24__umap-off__dims-easy3-hard3-medium3__optics-xi-6vals-fb1273__optics-ext-xi-dbscan_eps__sweep-mcs10-c59ca2-floor10-ms8-282082__9933bbd5ede5` |

XGBoost hard has no distinct downstream runner-up partition. The exported
downstream rows for that combination are duplicate partitions from nearby
hyperparameter settings, so adding them would not provide a new semantic
inspection target.

## Semantic Quality Evaluation

Interpretation conventions:

- Effects are on the interpretable model prediction scale for `ml_ade_log`; lower
  predicted values correspond to better trajectory prediction performance.
- A semantically good partition should give clusters with a clear feature-effect
  story, with the effect story reflected in the corresponding trajectory
  characteristics (`std_speed`, `max_speed`, `heading_change_per_sec`,
  `mean_acceleration`, `min_neighbor_distance`).
- The notes below use within-performance-group feature deviations. For example,
  `max_speed +0.8 z` means the cluster has a higher raw `max_speed` than typical
  members of that same easy, medium, or hard group.

### Overall Recommendation

| model | performance_group | best semantic choice | semantic verdict | reason |
| --- | --- | --- | --- | --- |
| GAM | easy | `tier_1` optics | strong | Three sizeable clusters form a coherent speed/turning taxonomy with low noise. |
| GAM | medium | `tier_1` optics | usable but weak | Captures a normal-medium group plus a small slow/high-turning tail; not a rich taxonomy. |
| GAM | hard | `structure_alt_1` optics, or `tier_1` for score-first use | strong tail pattern | All variants isolate the same erratic/high-acceleration hard tail; `structure_alt_1` gives slightly better coverage and lower noise. |
| XGBoost | easy | `structure_alt_2` optics | strong | Same two meaningful easy-case regimes as `tier_1`, with lower noise and less fragmentation. |
| XGBoost | medium | none clearly strong; `runner_up_1` if one must be inspected | weak | Meaningful hard-to-explain medium cases sit largely in noise; non-noise satellites are tiny. |
| XGBoost | hard | none | poor | The selected partition leaves about three quarters of hard cases as noise and only clusters small fragments. |

### GAM Additive Effects

**Easy.** The easy GAM partitions are semantically the cleanest part of the
shortlist. The main structure is stable across `tier_1`, `runner_up_1`, and
`runner_up_2`:

- A large high-speed/low-turning cluster (`max_speed` about `+0.8 z`,
  `heading_change_per_sec` about `-0.6 z`) has higher predicted and observed ADE
  within the easy group. Its effect signature is dominated by a higher
  `max_speed` contribution and a lower heading-change contribution.
- A large low-speed/high-turning cluster (`max_speed` about `-1.1 z`,
  `heading_change_per_sec` about `+0.9 z`) has lower predicted ADE. The lower
  `max_speed` effect more than offsets its higher heading-change effect.
- A small very-low-speed, low-variation cluster (`max_speed` about `-1.3 z`,
  `std_speed` about `-1.2 z`) has the lowest predicted and observed ADE and is
  plausibly an almost-stationary/simple-motion pattern.

Verdict by candidate:

| candidate | semantic verdict | notes |
| --- | --- | --- |
| `tier_1` optics | strong | Best balance: three interpretable clusters, 11% noise, and no obvious semantic redundancy. |
| `runner_up_1` optics | redundant/usable | Same story as `tier_1`, but 25% noise; no added semantic value. |
| `runner_up_2` hdbscan | strong but slightly redundant | Low noise and the same story, but the two smallest clusters have nearly identical very-low-speed profiles, so the four-cluster split overstates the semantic granularity. |
| `structure_alt_1` optics | lower priority | Same core story, but 41% noise and the slow/high-turning cluster shrinks strongly. |

**Medium.** The medium GAM candidates are semantically much thinner. They split a
large ordinary-medium cluster from a small low-speed/high-turning tail:

- The dominant cluster is close to the medium-group baseline and mostly says
  "typical medium case" rather than a distinct explanation pattern.
- The small tail has very low `max_speed` (`-2.4 z` to `-2.5 z`) and high
  `heading_change_per_sec` (`+2.3 z` to `+3.0 z`). Its lower predicted ADE is
  logically tied to the reduced `max_speed` effect, partly countered by a higher
  heading-change effect.

Verdict by candidate:

| candidate | semantic verdict | notes |
| --- | --- | --- |
| `tier_1` optics | usable but weak | Best medium choice because the tail is largest (`329` cases) and the profile is interpretable. |
| `runner_up_1` optics | weak | Same tail pattern, but only `156` cases; mostly a refinement of the dominant/noise boundary. |
| `runner_up_2` hdbscan | weak | Same tail pattern, only `103` cases, and substantially weaker raw-effect DBCV. |

**Hard.** The hard GAM candidates are semantically meaningful even though the
minority cluster is small. They isolate a high-risk hard-case tail:

- The dominant hard cluster has lower `std_speed` and slightly lower
  `mean_acceleration` than the hard-group average and has much lower predicted
  ADE than the tail.
- The small tail has high `std_speed` (about `+1.5 z`), high
  `mean_acceleration` (about `+1.6 z`), and closer neighbors. This is a coherent
  erratic/interactive trajectory pattern, and it also has higher observed and
  predicted ADE.
- The feature-effect relation is partly non-monotone: the poor tail is driven
  mainly by the `std_speed`, `max_speed`, and heading-change effects, while the
  `mean_acceleration` effect is not itself the adverse driver despite high raw
  acceleration. This weakens a simple one-feature narrative but not the cluster's
  overall semantic coherence.

Verdict by candidate:

| candidate | semantic verdict | notes |
| --- | --- | --- |
| `tier_1` optics | strong tail pattern | High DBCV and a coherent poor-performance tail, but the tail has only `48` cases. |
| `runner_up_1` optics | strong but redundant | Same pattern with `55` tail cases; no material semantic improvement over `tier_1`. |
| `runner_up_2` hdbscan | usable but lower priority | Same tail pattern with lower DBCV. |
| `structure_alt_1` optics | best semantic coverage | Same story, lower noise (`16%`), and a larger tail (`72` cases), at the cost of a lower score than `tier_1`. |

### XGBoost SHAP Effects

**Easy.** The easy XGBoost partitions recover the same core regimes as GAM, but
the best semantic split is the simpler two-cluster optics alternative:

- A high-speed/low-turning cluster (`max_speed` about `+0.8 z`,
  `heading_change_per_sec` about `-0.6 z`) has higher predicted and observed ADE
  within the easy group. It is characterized by a positive `max_speed` SHAP
  contribution and a negative heading-change contribution.
- A low-speed/high-turning cluster (`max_speed` about `-1.1 z`,
  `heading_change_per_sec` about `+0.6 z` to `+0.7 z`) has lower predicted and
  observed ADE. The negative `max_speed` SHAP contribution dominates the higher
  heading-change contribution.

Verdict by candidate:

| candidate | semantic verdict | notes |
| --- | --- | --- |
| `tier_1` optics | strong but noisier than needed | Clean two-cluster story, but 30% noise. |
| `runner_up_1` hdbscan | over-fragmented | The two large clusters are meaningful, but the remaining nine clusters are small satellites and 39% of cases are noise. |
| `runner_up_2` hdbscan | over-fragmented | Same as `runner_up_1`; the extra clusters do not produce distinct explanation patterns. |
| `structure_alt_1` hdbscan | usable but not preferred | Two large meaningful clusters plus small fragments; less clean than optics. |
| `structure_alt_2` optics | strongest semantic choice | Same interpretable two-cluster story as `tier_1`, lower noise (`20%`), and no small-cluster fragmentation. |

**Medium.** The medium XGBoost clusterings are weak as explanation taxonomies:

- The dominant non-noise cluster is close to the group baseline and mostly
  identifies ordinary medium cases with lower predicted ADE.
- The apparent low-speed/high-turning pattern exists, but it is tiny: `4` and
  `25` cases in `tier_1`, `28` cases in `runner_up_1`, or `54` plus `24` cases
  in `runner_up_2`.
- The noise bucket has the highest observed and predicted ADE and contains
  roughly 28% to 35% of the group. That means the most difficult medium cases are
  not organized into named clusters, which is a semantic failure for explanation
  purposes.

Verdict by candidate:

| candidate | semantic verdict | notes |
| --- | --- | --- |
| `tier_1` optics | weak | Better DBCV, but two of three clusters are too small (`25` and `4`) for robust semantic interpretation. |
| `runner_up_1` hdbscan | weak but simplest | Lower noise than the other medium XGBoost candidates and only one tiny tail, but still not a good explanation taxonomy. |
| `runner_up_2` hdbscan | weak | Splits the tiny low-speed/high-turning tail into multiple very small pieces and has low DBCV. |

**Hard.** The XGBoost hard partition should not be used as a semantic clustering
of hard regimes:

- About 75% of hard cases are noise, so the clustering does not explain most of
  the group.
- The non-noise clusters are small (`481`, `132`, `34`, `29`) and the raw-effect
  DBCV is very low.
- Some local patterns are interpretable - for example, a high `std_speed` SHAP
  cluster has high predicted ADE, and a lower-variation cluster has lower
  predicted ADE - but these are fragments rather than a coherent hard-case
  taxonomy.

Verdict by candidate:

| candidate | semantic verdict | notes |
| --- | --- | --- |
| `tier_1` optics | poor | Useful only for inspecting a few local hard-case archetypes, not for drawing a general semantic explanation of poor performance. |
