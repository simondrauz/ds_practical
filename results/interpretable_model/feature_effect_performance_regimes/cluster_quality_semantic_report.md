# Feature-effect performance-regime clustering report

Date: 2026-05-15

## Scope

This report evaluates the downstream-exported clusterings from four `ml_ade_log` feature-effect clustering runs. The analysis is restricted to the exported artifacts in each run directory: `cluster_scores.csv`, `cluster_catalog.csv`, `cluster_feature_effect_profiles.csv`, `cluster_assignments.csv`, and the exported cluster member files. Non-exported sweep candidates are not used for interpretation.

The performance groups use the original `ml_ade` scale. For the MI+VIF runs, the data contain 26,886 pedestrian trajectory rows: 15,823 easy, 8,320 medium, and 2,743 hard. The easy/medium and medium/hard boundaries are 0.388 and 1.198. For the VIF-only/no-collision runs, the data contain 26,122 rows: 15,348 easy, 8,094 medium, and 2,680 hard. The corresponding boundaries are 0.388 and 1.200.

## Method

Clustering quality was assessed using the exported conservative screen plus the raw effect-space DBCV, noise fraction, number of clusters, largest-cluster share, and smallest-cluster size. All selected downstream clusterings use OPTICS; HDBSCAN produced exported candidates in several runs but was not the selected candidate for any performance group.

Semantic quality was assessed by mapping each selected cluster's mean signed feature effects back to the underlying raw trajectory characteristics in `cluster_assignments.csv`. I used two reference frames:

- within-group offsets, because the question is whether clusters explain differences inside hard, medium, or easy;
- global offsets, because a hard-cluster feature can be only average within hard while still being high in absolute pedestrian-trajectory terms.

For semantic clarity, I treated a top-effect feature as narratively clear when the underlying raw feature distribution was visibly shifted in at least one reference frame. As a working diagnostic, I used robust median offsets: approximately 0.5 within-group IQR or 0.75 global IQR is a clear shift. This is a diagnostic threshold, not a domain threshold.

Noise was evaluated as a diagnostic group. When noise has coherent raw characteristics, it suggests a continuous or missed pattern, not a validated explanation cluster.

## Run Coverage

| run | clustering features | exported groups | main coverage issue |
| --- | --- | --- | --- |
| XGB / MI+VIF | `std_speed`, `max_speed`, `heading_change_per_sec`, `mean_acceleration` | easy, medium, hard | hard has very high selected noise |
| GAM / MI+VIF | `std_speed`, `max_speed`, `heading_change_per_sec`, `mean_acceleration` | easy, medium, hard | medium is dominated by one large cluster |
| XGB / VIF only | `std_speed`, `heading_change_per_sec`, `min_neighbor_distance`, `mean_acceleration`, `scene_density_VEHICLE`, `scene_num_VEHICLE` | hard only | no exported easy or medium clusterings |
| GAM / VIF only | `heading_change_per_sec`, `mean_acceleration`, `min_neighbor_distance`, `std_speed`, `scene_num_VEHICLE` | medium, hard | no exported easy clustering; medium is weak |

The smaller MI+VIF feature set gives better regime coverage. The larger VIF-only/no-collision feature set does not improve semantic coverage; it mostly produces hard-regime clusters and remains dominated by speed-variation and heading-change explanations.

## Candidate Quality Summary

Values are computed over exported candidates only.

| run | group | algorithm | candidates | DBCV med/max | noise med/range | clusters med/range | largest med/range | n group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAM / MI+VIF | easy | hdbscan | 44 | 0.238/0.345 | 0.143 (0.091-0.255) | 3.5 (3-6) | 0.535 (0.532-0.545) | 15823 |
| GAM / MI+VIF | easy | optics | 57 | 0.070/0.401 | 0.499 (0.249-0.749) | 6.0 (4-11) | 0.241 (0.133-0.536) | 15823 |
| GAM / MI+VIF | hard | hdbscan | 20 | 0.080/0.122 | 0.653 (0.281-0.712) | 2.0 (2-4) | 0.238 (0.171-0.706) | 2743 |
| GAM / MI+VIF | hard | optics | 173 | 0.088/0.194 | 0.694 (0.250-0.749) | 7.0 (2-20) | 0.187 (0.022-0.744) | 2743 |
| GAM / MI+VIF | medium | optics | 8 | 0.136/0.136 | 0.150 (0.150-0.150) | 3.0 (3-3) | 0.843 (0.843-0.843) | 8320 |
| GAM / VIF only | hard | hdbscan | 14 | 0.074/0.128 | 0.571 (0.194-0.591) | 2.0 (2-3) | 0.355 (0.349-0.784) | 2680 |
| GAM / VIF only | hard | optics | 114 | 0.074/0.367 | 0.498 (0.147-0.750) | 5.0 (2-20) | 0.406 (0.026-0.837) | 2680 |
| GAM / VIF only | medium | optics | 16 | 0.097/0.114 | 0.225 (0.150-0.300) | 2.0 (2-2) | 0.770 (0.695-0.845) | 8094 |
| XGB / MI+VIF | easy | hdbscan | 60 | 0.220/0.301 | 0.334 (0.288-0.365) | 5.0 (3-8) | 0.436 (0.408-0.491) | 15823 |
| XGB / MI+VIF | easy | optics | 131 | 0.047/0.411 | 0.350 (0.200-0.749) | 3.0 (2-15) | 0.324 (0.147-0.435) | 15823 |
| XGB / MI+VIF | hard | hdbscan | 11 | 0.040/0.045 | 0.734 (0.711-0.746) | 3.0 (3-4) | 0.166 (0.166-0.175) | 2743 |
| XGB / MI+VIF | hard | optics | 90 | 0.054/0.091 | 0.699 (0.200-0.748) | 3.0 (2-4) | 0.203 (0.172-0.799) | 2743 |
| XGB / MI+VIF | medium | hdbscan | 6 | 0.037/0.038 | 0.246 (0.244-0.246) | 2.0 (2-3) | 0.746 (0.746-0.746) | 8320 |
| XGB / MI+VIF | medium | optics | 80 | 0.335/0.397 | 0.275 (0.200-0.400) | 2.0 (2-2) | 0.721 (0.598-0.793) | 8320 |
| XGB / VIF only | hard | hdbscan | 34 | 0.125/0.192 | 0.524 (0.493-0.553) | 2.0 (2-2) | 0.458 (0.422-0.481) | 2680 |
| XGB / VIF only | hard | optics | 300 | 0.187/0.293 | 0.498 (0.299-0.748) | 2.0 (2-4) | 0.484 (0.143-0.671) | 2680 |

## Selected Clusterings

| run | group | alg | clusters | noise | DBCV | largest | smallest n | quality read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB / MI+VIF | easy | optics | 2 | 0.300 | 0.411 | 0.364 | 5323 | good separation; interpretable split |
| XGB / MI+VIF | medium | optics | 2 | 0.400 | 0.397 | 0.598 | 19 | good DBCV, but second cluster is tiny and noise is high |
| XGB / MI+VIF | hard | optics | 4 | 0.748 | 0.091 | 0.172 | 31 | weak clustering; most hard cases are noise |
| GAM / MI+VIF | easy | optics | 4 | 0.324 | 0.401 | 0.536 | 633 | good DBCV; one large cluster plus slow/stationary subclusters |
| GAM / MI+VIF | medium | optics | 3 | 0.150 | 0.136 | 0.843 | 27 | weak/moderate; dominated by one cluster |
| GAM / MI+VIF | hard | optics | 4 | 0.699 | 0.194 | 0.159 | 126 | moderate semantic signal, weak due high noise |
| XGB / VIF only | hard | optics | 2 | 0.450 | 0.293 | 0.541 | 25 | moderate; one broad cluster, one tiny outlier cluster |
| GAM / VIF only | hard | optics | 2 | 0.148 | 0.367 | 0.837 | 40 | strongest numeric hard DBCV, but structurally broad-plus-outlier |
| GAM / VIF only | medium | optics | 2 | 0.300 | 0.114 | 0.695 | 41 | weak; low DBCV and a tiny second cluster |

## Performance Within Selected Clusters

Cluster-level `ml_ade` is important for interpreting broad-plus-extreme splits. The original report interpretation used the performance group as the conditioning frame; the additional check below shows whether selected clusters also separate severity within the group.

| run | group | key within-group `ml_ade` pattern |
| --- | --- | --- |
| XGB / MI+VIF | easy | cluster 0 is substantially easier than the group median (`ml_ade` median 0.083 vs 0.167), cluster 1 is moderately harder within easy (0.199), and noise is hardest within easy (0.238). |
| XGB / MI+VIF | medium | the named clusters are not the higher-error part; noise has the highest median `ml_ade` (0.726 vs group median 0.602). This supports the earlier interpretation that the medium speed-variation story is mostly continuous/noisy rather than cleanly clustered. |
| XGB / MI+VIF | hard | non-noise clusters are mostly lower-error hard cases (cluster medians 1.500 to 1.942 vs group median 1.782), while noise is higher-error (1.934). The selected hard clusters should not be read as the main extreme-hard explanation. |
| GAM / MI+VIF | easy | the small slow/stationary clusters are the easiest cases (0.023 to 0.068 vs group median 0.167); the large cluster is harder within easy (0.218). |
| GAM / MI+VIF | medium | noise again has the higher `ml_ade` median (0.721 vs 0.602), while the named clusters are at or below the group median. |
| GAM / MI+VIF | hard | cluster 3 is both semantically extreme and higher-error (2.457 vs group median 1.782). This is the clearest MI+VIF hard subgroup where feature-effect extremity maps to worse Trajectron++ performance. |
| XGB / VIF only | hard | the tiny cluster 1 is a true extreme-hard subgroup (`ml_ade` median 3.384 vs group median 1.783), while the large cluster is lower-error hard (1.571) and noise is also high-error (2.265). |
| GAM / VIF only | hard | the tiny cluster 1 and noise are both high-error hard cases (2.442 and 2.410 vs group median 1.783), while the dominant cluster is lower-error hard (1.706). |
| GAM / VIF only | medium | noise has the higher medium-regime error (0.712 vs group median 0.601); the two named clusters do not separate severity much. |

This changes the hard-regime conclusion slightly: the VIF-only selected hard clusterings are useful as extreme-hard detectors, but mostly because of a tiny cluster plus coherent high-error noise. XGB / MI+VIF hard is not a good discrete explanation of the worst hard cases, because the highest-error mass remains outside the non-noise clusters.

## Whole Performance-Group Profiles

When a selected clustering is noisy, dominated by one large cluster, or missing for a performance group, the whole performance group still gives useful regime-level evidence. I computed the median sum of signed feature effects within each group, the share of rows with positive total feature effect, and the strongest per-feature signed effects. Feature-effect values are on the meta-model target scale (`ml_ade_log`); positive values increase predicted Trajectron++ error and negative values decrease it.

| run | group | whole-group effect profile | value added beyond clusters |
| --- | --- | --- | --- |
| XGB / MI+VIF | easy | median total effect -0.195; only 6.4% positive. `std_speed` is mostly negative (median -0.095, 6% positive, raw `std_speed` -0.28 global IQR), and `heading_change_per_sec` is also mostly negative (median -0.055). | Confirms that the easy group is broadly a low-error regime, not just two selected clusters. It also explains why the noise rows are still easy even when they are not cleanly clustered. |
| XGB / MI+VIF | medium | median total effect +0.057; 66.5% positive. `max_speed` is consistently positive (median +0.067, 90% positive), while `std_speed` and heading effects are mixed. | Adds a better medium-regime narrative than the selected clusters: medium cases are a broad weakly positive difficulty regime, with speed-related effects but little clean substructure. |
| XGB / MI+VIF | hard | median total effect +0.525; 99.5% positive. `std_speed` (median +0.286), `heading_change_per_sec` (+0.130), and `max_speed` (+0.118) are all strongly positive; raw `std_speed` is +2.00 global IQR. | This is the clearest hard-regime explanation when clustering fails: high speed variation plus turning/high speed broadly raises predicted error across almost the whole hard group. |
| GAM / MI+VIF | hard | median total effect +0.098; 61.2% positive. `mean_acceleration` (+0.189) and `heading_change_per_sec` (+0.071) are positive, but the broad `std_speed` median is negative despite high raw `std_speed`. | Supports a hard-regime acceleration/turning story, but it weakens any simple GAM-wide claim that speed variation itself always increases predicted error. The clearest GAM speed-variation story remains the hard outlier cluster. |
| XGB / VIF only | medium | median total effect +0.098; 83.4% positive. `std_speed` (+0.044) and `heading_change_per_sec` (+0.033) are positive, while `min_neighbor_distance` is small (+0.007). | Fills a gap because no medium clustering was exported for this run. It says the larger feature set still gives a speed/turning medium-regime profile, not a strong neighbor-distance or vehicle-context profile. |
| XGB / VIF only | hard | median total effect +0.622; 100% positive. `std_speed` (+0.376), `heading_change_per_sec` (+0.112), and `mean_acceleration` (+0.050) are the leading positive effects. | Strengthens the hard-regime conclusion: even apart from the tiny extreme cluster, the whole hard group is a high positive-effect regime driven by speed variation and turning. |
| GAM / VIF only | hard | median total effect +0.013; 50.9% positive. `heading_change_per_sec` (+0.266) and `mean_acceleration` (+0.486) are strongly positive, while `std_speed` is negative at the whole-group median. | Adds support for turning/acceleration in hard cases, but not for a broad positive net GAM profile. The selected tiny cluster and high-error noise are more informative than the whole hard group here. |

The whole-group profile does not add much for the GAM medium groups: the selected clusters are weak, but the total feature-effect medians are also negative or close to neutral, and the higher-error medium rows mainly appear in noise. I would not turn those into medium-regime narratives.

## Semantic Assessment By Run

### XGB / MI+VIF

This is the best all-around run for explanation narratives because it exports all three performance groups and the feature set is trajectory-centered.

Easy selected clustering: quality is strong. The two non-noise clusters are both large. One cluster has negative effects for `heading_change_per_sec` and `std_speed` with near-typical raw speed variation and below-global heading change. The other has negative effects for `max_speed` and `std_speed` but a positive effect for `heading_change_per_sec`; its trajectories are very low-speed (`max_speed` about 1.12 global IQR below the global median) but high-turning (`heading_change_per_sec` about 1.42 global IQR above the global median). This produces a useful easy-regime narrative: some easy trajectories are straightforward because they are slow despite turning, while another large group remains easy because heading changes and speed variation lower the modelled error contribution.

Medium selected clustering: numeric separation is good, but semantic quality is only moderate. The main cluster covers about 60% of medium rows and has small positive `max_speed` effects with otherwise weak raw shifts. The second cluster has only 19 rows; it is low-speed and very high-turning. The noise group, not the tiny second cluster, carries a more obvious medium-regime story: high `std_speed` and mildly high heading change with positive feature effects. This suggests a continuum of speed-variation difficulty rather than a clean discrete medium pattern.

Hard selected clustering: clustering quality is weak because 74.8% of hard rows are noise and DBCV is only 0.091. The non-noise clusters do have plausible narratives: the clearest small clusters have positive `std_speed` and `heading_change_per_sec` effects, and the underlying trajectories have globally high speed variation and often high turning. However, the noise group has almost the same story (`std_speed` about 2.42 global IQR above the global median) and has the higher median `ml_ade`. The hard regime is therefore better described as a broad hard continuum with small outlier pockets, not as four stable explanation patterns.

### GAM / MI+VIF

This run gives complete group coverage, but the GAM effect profiles are less granular. The exported clusters are often dominated by `std_speed` and `mean_acceleration`.

Easy selected clustering: DBCV is high, and the raw narratives are acceptable but less differentiated than XGB. All main clusters have strongly negative `std_speed` effects and positive `mean_acceleration` effects. The smaller clusters are very low-speed or stationary, so they support an easy-regime narrative of slow or stationary pedestrians. The largest cluster is only mildly shifted, so the overall story is broad rather than sharply segmented.

Medium selected clustering: this is weak semantically. The largest cluster contains 84.3% of medium rows and has only small raw shifts. The two additional clusters contain 34 and 27 rows and mainly isolate low-`max_speed` cases. The noise group is more semantically interesting than the clusters because it has high `std_speed` (about 1.86 within-group IQR and 2.13 global IQR above the medians), but it is not a validated cluster. This is not a good basis for multiple medium-regime explanation patterns.

Hard selected clustering: the clustering is noisy, but at least one narrative is clear. One hard cluster has positive `std_speed`, positive `heading_change_per_sec`, and high `mean_acceleration` effects; its underlying trajectories have very high speed variation, many turns, and high acceleration. Other clusters split combinations of high turning, low speed, and acceleration direction, but 69.9% noise limits confidence. The useful takeaway is a robust hard outlier pattern: high speed variation plus turning/acceleration.

### XGB / VIF Only, No Collision

Only hard-regime candidates were exported, so this run is not useful for a full easy/medium/hard explanation comparison.

The selected hard clustering has moderate DBCV but 45.0% noise. It splits a large hard cluster from a tiny high-variation outlier cluster. The large cluster has positive `std_speed` and `heading_change_per_sec` effects and globally high speed variation. The small cluster is much more extreme: `std_speed` is about 4.83 global IQR above the median, with high heading change and high acceleration. The added VIF-only features (`min_neighbor_distance`, vehicle count/density) do not drive the main selected narratives. Semantically, this is a hard-regime speed-variation story, not a richer scene-interaction story.

### GAM / VIF Only, No Collision

The selected hard clustering has the best raw DBCV among hard selected candidates and low noise relative to the other hard runs, but it is structurally imbalanced: 83.7% of hard rows are in one cluster and only 40 rows are in the second. This is a good outlier detector, not a good multi-pattern explanation clustering.

Hard selected clustering: the small cluster is very clear, with high `std_speed`, high heading change, and high acceleration. The noise group has the same direction and is also semantically coherent. The large cluster has globally high speed variation but is typical or below-typical within the hard group. The right narrative is therefore: most hard cases are broadly high-variation, while a small extreme subgroup has especially high variation, turns, and acceleration.

Medium selected clustering: quality and semantic value are weak. The largest cluster covers 69.5% of rows, the second has only 41 rows, and DBCV is 0.114. The second cluster has high `std_speed` but low heading change, while the noise group has both high `std_speed` and higher heading change. This looks more like a residual split plus noise than a clean medium-regime explanation pattern.

## Cross-run Findings

The smaller MI+VIF feature set is more useful for this analysis. It gives exported clusterings for all performance groups and keeps the explanations tied to direct pedestrian trajectory characteristics. The VIF-only/no-collision feature set adds neighbor and scene-vehicle variables, but the exported narratives still mostly reduce to `std_speed` and `heading_change_per_sec`. It also loses easy-regime exports entirely.

Hard-regime clustering is the main limitation across runs. The hard group has coherent semantic content, especially high speed variation, turning, and acceleration, but the structure is often continuous rather than discretely clustered. This shows up as high noise in XGB / MI+VIF and GAM / MI+VIF, or as a dominant broad cluster plus tiny outlier in the VIF-only runs.

Medium-regime narratives are less stable than easy or hard. XGB / MI+VIF has good DBCV for medium, but the second selected cluster is tiny and noise contains much of the speed-variation story. GAM medium runs are dominated by one large cluster and should not be used to claim several distinct medium explanation patterns.

Easy-regime narratives are strongest in the MI+VIF runs. XGB / MI+VIF provides the cleanest split: slow/high-turning trajectories can still be easy because low speed offsets turning difficulty, while another large easy cluster has generally low difficulty effects. GAM / MI+VIF supports a simpler slow or stationary pedestrian narrative.

Noise is informative but should not be renamed as a cluster. In hard and some medium runs, noise often has coherent high-variation trajectory characteristics. That means the feature-effect space likely contains gradients or boundary cases that the conservative clustering does not represent as stable clusters. It is useful evidence for "hard because speed/heading changes are high", but not enough for a discrete explanation-pattern claim.

## Recommended Use

Use XGB / MI+VIF as the primary source for cross-regime explanation patterns. It is the only run with a good combination of coverage, trajectory-centered features, and interpretable easy/medium/hard behavior. For hard-regime severity specifically, combine it with the VIF-only hard results because those selected tiny clusters have clearly higher `ml_ade`.

Use GAM / MI+VIF as a robustness check, especially for the hard high-speed-variation plus turning/acceleration pattern. Avoid presenting its medium clusters as multiple strong narratives.

Use VIF-only/no-collision runs only as hard-regime supplementary evidence. They support the importance of speed variation and turning but do not add a convincing neighbor-distance or vehicle-context narrative in the exported selected clusterings.

For the final paper or presentation, label hard-regime cluster narratives with confidence qualifiers. The best wording is not "the hard group consists of four explanation patterns"; it is closer to "hard predictions are dominated by a continuous high-variation/turning regime, with small extreme subgroups that clustering can isolate."

When selected clusterings do not expose clear groups, inspect the whole performance group as a regime-level profile. A useful fallback is to report the group's signed feature-effect medians, interquartile ranges, sign shares, and raw-characteristic offsets relative to the full pedestrian distribution. This should be presented as a broad regime explanation rather than a cluster narrative.

Density clustering is not guaranteed to return one large cluster when a whole performance group has a single coherent structure. If the feature-effect distribution is unimodal or forms a continuous gradient, HDBSCAN/OPTICS may produce no stable cluster, mark boundary cases as noise, or return one dominant cluster plus small edge clusters. The conservative export filters also make this more likely because they avoid degenerate or overly dominant clusterings. Therefore, whole-group feature-effect inspection is not redundant; it is the right diagnostic when clustering indicates "no strong substructure."
