# Eye-Tracking Gaze Annotation Comparison Rules

## 1. Data Selection

### 1.1 Use the Original Eye-Tracking Annotations

For the comparison, use the original eye-tracking gaze annotations from both annotators directly.

### 1.2 Duration Threshold for Formal Comparison

A duration threshold of **100 ms** is applied when determining which gaze intervals enter the formal comparison.

- Gaze intervals with a duration of **less than 100 ms**:
  - remain in the original annotation data;
  - are **excluded from the formal comparison**.

- Gaze intervals with a duration of **100 ms or longer**:
  - are **included in the formal comparison**.

Therefore, only gaze intervals with **duration ≥ 100 ms** are used in the formal comparison and merging procedure described below.

### 1.3 Rationale for the 100 ms Threshold

The **100 ms** threshold is used as a conservative minimum-duration criterion for deciding which annotated gaze intervals enter the formal comparison.

Manor and Gordon (2003) systematically examined temporal thresholds for ocular fixation below the commonly used 200 ms threshold. They reported that a **100 ms temporal threshold effectively discriminated fixations from other oculomotor activity** and was consistent with physiological and visuocognitive models. They also noted that thresholds substantially below 100 ms increasingly risk including samples associated with saccadic activity rather than stable fixation behavior.

In the present study, the 100 ms threshold is therefore used as a **comparison-entry criterion**, rather than as an automatic deletion rule for the original annotation data. Intervals shorter than 100 ms are retained in the original annotations but are not included in the formal comparison and merging procedure.

ISO 15007:2020 is additionally used as the general methodological reference for the measurement and analysis of driver visual behaviour and glance-related measures in driving studies.

---

## 2. Formal Comparison and Merging Rules

For gaze intervals from the two annotators that meet the formal comparison criterion (**duration ≥ 100 ms**), the comparison is performed based on temporal overlap structure.

For notation:

- `H` = Huilin's gaze annotation
- `J` = Jakob's gaze annotation

The formal comparison distinguishes between:

1. one-to-one overlap cases;
2. complex overlap cases;
3. no-overlap cases.

### 2.1 One-to-One Overlap Cases: `H1 ↔ J1`

This category includes both:

- complete containment;
- partial overlap.

For all `H1 ↔ J1` cases, the same final-boundary rule is used.

#### 2.1.1 Diagnostic Metrics

For each one-to-one pair, calculate:

```text
overlap_duration
```

```text
union_duration
```

```text
disagreement_duration
= union_duration - overlap_duration
```

```text
IoU
= overlap_duration / union_duration
```

where:

- `overlap_duration` is the duration for which both annotators label gaze;
- `union_duration` is the total duration covered by at least one of the two gaze intervals;
- `disagreement_duration` is the total time labeled as gaze by only one annotator;
- `IoU` describes the degree of temporal agreement between the two intervals.

#### 2.1.2 Final Consensus Gaze

For all `H1 ↔ J1` cases, the final gaze interval is calculated using the mean of the two annotated boundaries:

```text
start_final
= (H_start + J_start) / 2
```

```text
end_final
= (H_end + J_end) / 2
```

#### 2.1.3 Large-Disagreement Flag

A one-to-one case is flagged for review if:

```text
disagreement_duration > 2000 ms
AND
IoU < 0.20
```

The 2000 ms threshold corresponds to the 90th percentile of disagreement duration across all one-to-one cases. The IoU condition excludes cases in which both annotators labeled the same gaze and only the start or end boundaries differ, which can also produce a large disagreement duration.

Flagged cases are **not automatically rejected or changed**. Instead:

- the case is flagged;
- IoU and the specific start/end boundary differences are retained as supporting diagnostic information;
- the flagged case is checked in the eye-tracking video.

The **2000 ms threshold is a review-flag threshold**, not a validity threshold for gaze duration.

---

### 2.2 Complex Overlap Cases

Complex cases include overlap clusters such as:

```text
H1 ↔ J2
H2 ↔ J1
H1 ↔ J3
H3 ↔ J1
H2 ↔ J2
H3 ↔ J3
...
```

For these cases, all gaze intervals from each annotator within the same overlap cluster are treated as temporal sets.

Let:

```text
H = union of all Huilin gaze intervals in the cluster
```

```text
J = union of all Jakob gaze intervals in the cluster
```

#### 2.2.1 Cluster-Level Diagnostic Metrics

For every complex overlap cluster, calculate:

```text
agreement_duration
= duration(H ∩ J)
```

```text
cluster_union_duration
= duration(H ∪ J)
```

```text
disagreement_duration
= cluster_union_duration - agreement_duration
```

```text
cluster_IoU
= agreement_duration / cluster_union_duration
```

`cluster_union_duration` is the total amount of time for which **at least one annotator** labels gaze.

#### 2.2.2 Single-Interval Average-Boundary Candidate

For **all complex overlap cases**, regardless of whether the two annotators have the same or different numbers of gaze intervals, also calculate a candidate result in which the entire complex cluster is represented as **one continuous gaze interval**.

The start boundary is calculated as:

```text
single_gaze_start
= (H_first_start + J_first_start) / 2
```

The end boundary is calculated as:

```text
single_gaze_end
= (H_last_end + J_last_end) / 2
```

where:

- `H_first_start` is the start time of Huilin's first gaze interval in the cluster;
- `J_first_start` is the start time of Jakob's first gaze interval in the cluster;
- `H_last_end` is the end time of Huilin's last gaze interval in the cluster;
- `J_last_end` is the end time of Jakob's last gaze interval in the cluster.

The corresponding candidate duration is:

```text
single_gaze_duration
= single_gaze_end - single_gaze_start
```

This candidate is calculated and retained **for reference only**. It is not used to replace or determine the segmentation-preserving consensus result at the current comparison stage. It may be used later for comparison or modeling-related evaluation.

#### 2.2.3 Complex Cases with Different Numbers of Gaze Intervals

Examples include:

```text
H1 ↔ J2
H2 ↔ J1
H1 ↔ J3
H3 ↔ J1
...
```

If the two annotators have different numbers of gaze intervals within the same overlap cluster, the current segmentation-preserving rule is:

> **Retain the annotation from the annotator with the larger number of gaze intervals.**

For these clusters, the segmentation-preserving consensus result is taken from the annotator with the larger number of gaze intervals.

#### 2.2.4 Complex Cases with the Same Number of Gaze Intervals

Examples include:

```text
H2 ↔ J2
H3 ↔ J3
```

##### 2.2.4.1 Clear Ordered One-to-One Matching

If the intervals can be matched clearly in temporal order, for example:

```text
H1 ↔ J1
H2 ↔ J2
```

then each matched pair is processed separately using the average-boundary rule.

For each matched pair:

```text
start_segment
= (H_start + J_start) / 2
```

```text
end_segment
= (H_end + J_end) / 2
```

For example, in an `H2 ↔ J2` cluster:

```text
final_segment_1:
start = average(H1_start, J1_start)
end   = average(H1_end,   J1_end)

final_segment_2:
start = average(H2_start, J2_start)
end   = average(H2_end,   J2_end)
```

This result preserves the temporal segmentation.

For these clusters, the segment-wise average-boundary result is used as the segmentation-preserving consensus result.

##### 2.2.4.2 Unclear Matching

If a clear ordered one-to-one correspondence cannot be established:

- flag the cluster for later review / discussion;
- do not automatically merge the whole cluster;
- do not automatically select one annotator's segmentation;

#### 2.2.5 Large-Disagreement Flag

A complex cluster is flagged for review if:

```text
disagreement_duration > 4000 ms
AND
cluster_IoU < 0.20
```

A higher disagreement threshold is used than for one-to-one cases, because disagreement accumulates over several gaze intervals within a cluster. The 4000 ms threshold is close to the 90th percentile of disagreement duration across all complex clusters (approximately 4300 ms). As for one-to-one cases, flagged clusters are **not automatically rejected or changed**.

---

### 2.3 No-Overlap Cases

If a gaze interval from one annotator has no temporal overlap with any gaze interval from the other annotator, it is treated as a **no-overlap / single-annotator gaze event**.

These gaze intervals should be **directly included in the final consensus gaze dataset** without merging.

For each no-overlap gaze interval:

- retain the original start and end time from the annotator who identified the gaze;
- retain the original duration;
- add a flag indicating that the interval is a **no-overlap gaze**;
- retain the source annotator identity for traceability.

For example, the final output may include:

```text
comparison_type = no_overlap
source_annotator = Huilin / Jakob
```

This allows no-overlap gaze events to remain in the final consensus dataset while remaining distinguishable from gaze intervals supported by both annotators.

---
### 2.4 Review of Flagged Cases

All flagged cases (23 one-to-one cases and 6 complex clusters) were checked in the eye-tracking video. The disagreements reflect differences in annotation definitions rather than annotation errors, so the rule-based result is retained for all flagged cases. The main sources of disagreement are summarized in the results file.

---

## 3. Summary of the Current Comparison Logic

For all gaze intervals with **duration ≥ 100 ms**:

### 3.1 One-to-One Overlap: `H1 ↔ J1`

- calculate overlap, union, disagreement duration, and IoU;
- use average start and average end as the final consensus boundaries;
- flag cases with `disagreement_duration > 2000 ms` and `IoU < 0.20` for review;
- use IoU and boundary differences as supporting diagnostic information.

### 3.2 Complex Overlap

For every complex cluster:

- calculate `agreement_duration`, `cluster_union_duration`, `disagreement_duration`, and `cluster_IoU`;
- additionally calculate `single_gaze_start`, `single_gaze_end`, and `single_gaze_duration` as a **reference-only single-interval candidate**;
- flag clusters with `disagreement_duration > 4000 ms` and `cluster_IoU < 0.20` for review.

If the two annotators have different numbers of gaze intervals:

- retain the annotation from the annotator with the larger number of gaze segments as the segmentation-preserving consensus result.

If the two annotators have the same number of gaze intervals:

- if clear ordered one-to-one matching is possible, use the segment-wise average-boundary result as the segmentation-preserving consensus result;
- if clear matching is not possible, flag the cluster for later review / discussion.

### 3.3 No Overlap

- directly include the original single-annotator gaze interval in the final consensus gaze dataset;
- do not merge it with another interval;
- retain a `no_overlap` flag and the source annotator identity for traceability.

Gaze intervals with **duration < 100 ms** remain in the original annotation dataset but do not participate in the formal comparison.

---

## 4. References

- Manor, B. R., & Gordon, E. (2003). *Defining the temporal threshold for ocular fixation in free-viewing visuocognitive tasks*. Journal of Neuroscience Methods, 128(1–2), 85–93. DOI: `10.1016/S0165-0270(03)00151-1`.
- ISO 15007:2020. *Road vehicles — Measurement and analysis of driver visual behaviour with respect to transport information and control systems*. International Organization for Standardization.
