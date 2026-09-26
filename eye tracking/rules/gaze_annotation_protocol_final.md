# Gaze Annotation Protocol for Eye-tracking Video Coding

## 1. Purpose

This annotation protocol defines how to code `Gaze_to_initiator` events from eye-tracking videos. The goal is to ensure that both annotators apply the same AOI definition and annotation criteria before comparing and merging the two annotated datasets.

The current annotation focuses on whether the ego driver looks at the interacting initiator vehicle during each driving scenario.

## 2. Annotated Behavior

The annotated behavior is:

```text
Gaze_to_initiator
```

This behavior should be coded as a **state event**, with one `START` and one `STOP` for each continuous gaze episode.

A gaze episode starts when the gaze point begins to fall on the defined initiator AOI, and ends when the gaze point clearly leaves the initiator AOI.

## 3. Annotation Parameters and Event Markers

In addition to `Gaze_to_initiator`, the annotation data include temporal markers used to identify the beginning of runs/conditions and individual scenarios.

### 3.1 `Scenario_N_Start`

`Scenario_N_Start` corresponds to the moment when the fog clears at the beginning of each scenario.

The following markers may therefore be annotated:

```text
Scenario_1_Start
Scenario_2_Start
Scenario_3_Start
Scenario_4_Start
```

Each marker represents the visible start of the corresponding scenario in the eye-tracking video.

### 3.2 Condition Start / Run Start

The condition start or run start corresponds to the moment when the eye-tracking video suddenly becomes bright at the beginning of the run.

This visually identifiable transition should be used to determine the corresponding condition/run start marker in the annotation data.

## 4. Repeated Runs

If multiple repeated runs exist for the same participant and condition, only the run containing the complete set of all four scenarios should be annotated and retained for subsequent comparison and analysis.

A complete run should contain:

```text
Scenario_1
Scenario_2
Scenario_3
Scenario_4
```

Therefore:

- gaze events should be annotated only for the repeated run that contains the complete `Scenario_1`–`Scenario_4` sequence;
- gaze events from incomplete repeated runs do not need to be annotated;
- incomplete repeated runs should not enter the subsequent gaze comparison and analysis.

If only one run exists for a participant-condition, it should not be excluded solely because the beginning of the run is missing, provided that the remaining scenarios needed for the analysis can still be identified reliably.

## 5. AOI Definition for `Gaze_to_initiator`

Following ISO 15007:2020, an AOI is treated as a predefined area within the visual scene. In this study, the AOI for `Gaze_to_initiator` is defined as:

```text
the initiator vehicle body and its immediate surrounding area
```

The AOI should be defined before annotation and should not be interpreted as the entire lane, the target lane, or the general forward roadway. ISO 15007 provides the driver visual-behaviour framework and notes that AOIs are used for glance-related measures, but the exact vehicle-specific AOI boundary needs to be defined by the study protocol.

## 6. Valid Cases

A gaze interval should be coded as `Gaze_to_initiator` when one of the following conditions is satisfied:

1. The gaze point is clearly located on the initiator vehicle body.
2. The gaze point overlaps with the initiator vehicle.
3. In borderline cases, the gaze point is immediately adjacent to the initiator vehicle and can reasonably be interpreted as being directed toward the initiator.

The best case is when the gaze point is completely located on the initiator vehicle. The acceptable case is when the gaze point overlaps with the initiator vehicle. The borderline case is when the gaze point is immediately adjacent to the initiator vehicle, but this should be checked carefully.

Only gaze intervals that satisfy the valid-case criteria above should be coded as `Gaze_to_initiator`. Intervals with clearly unreliable gaze signals, such as severe gaze drift, systematic misalignment, or missing gaze points, should not be coded.

## 7. References

- ISO 15007:2020. *Road vehicles — Measurement and analysis of driver visual behaviour with respect to transport information and control systems*. Standard identifier: ISO 15007:2020.
