# Eye-tracking Annotation and Driving Data Synchronization Rules

## 1. Synchronization Anchor

For each participant and condition, the eye-tracking annotation should be synchronized with the corresponding driving data using predefined synchronization anchors.

Two time references are involved:

1. the **eye-tracking anchor**, defined on the eye-tracking annotation timeline;
2. the **driving-data anchor**, defined on the ASC `TimeMS` timeline.

The way these two anchors are obtained depends on whether `RunN_Start` is available.

### 1.1 Events with an identifiable Run Start

If `RunN_Start` can be reliably identified in the eye-tracking annotation, `RunN_Start` is used as the eye-tracking synchronization anchor.

The two annotators' original `RunN_Start` times are retained. The final eye-tracking anchor time is calculated as:

```text
Final Eye-Tracking Anchor Time
= (Run Start Time_A + Run Start Time_B) / 2
```

The corresponding driving-data anchor is the first valid frame of the corresponding ASC file:

```text
ASC Anchor TimeMS
= first valid ASC TimeMS
```

### 1.2 Events without an identifiable Run Start

If `RunN_Start` cannot be reliably identified, a predefined `Scenario_Start` is used as the eye-tracking synchronization anchor.

For these special cases, the eye-tracking anchor and the driving-data anchor are derived from two different time sources:

- the **Final Eye-Tracking Anchor Time** is calculated from the two annotators' `Scenario_Start` times on the eye-tracking annotation timeline;
- the **ASC Anchor TimeMS** is determined from the two annotators' independently identified `Scenario_Start` times in  POV videos.

#### 1.2.1 P06 AV0

For `P06 AV0`:

- `RunN_Start` is unavailable;
- `Scenario_1_Start` cannot be reliably identified in the eye-tracking video;
- therefore, `Scenario_2_Start` is used.

##### Eye-tracking anchor

The final eye-tracking anchor time is calculated from the two annotators' `Scenario_2_Start` times on the eye-tracking annotation timeline:

```text
Final Eye-Tracking Anchor Time
= (Scenario_2_Start_eye_time_A + Scenario_2_Start_eye_time_B) / 2
```

##### Driving-data anchor

The two annotators independently identify `Scenario_2_Start` from their respective POV videos.

The mean POV scenario-start time is calculated as:

```text
Mean Scenario_2_Start POV Time
= (Scenario_2_Start_POV_time_A + Scenario_2_Start_POV_time_B) / 2
```

The driving-data anchor is then defined as the ASC frame whose `TimeMS` is closest to this mean POV time:

```text
ASC Anchor TimeMS
= ASC TimeMS closest to Mean Scenario_2_Start POV Time
```

#### 1.2.2 P21 HV1

For `P21 HV1`:

- `RunN_Start` is unavailable;
- `Scenario_1_Start` is used.

##### Eye-tracking anchor

The final eye-tracking anchor time is calculated from the two annotators' `Scenario_1_Start` times on the eye-tracking annotation timeline:

```text
Final Eye-Tracking Anchor Time
= (Scenario_1_Start_eye_time_A + Scenario_1_Start_eye_time_B) / 2
```

##### Driving-data anchor

The two annotators independently identify `Scenario_1_Start` from their respective POV videos.

The mean POV scenario-start time is calculated as:

```text
Mean Scenario_1_Start POV Time
= (Scenario_1_Start_POV_time_A + Scenario_1_Start_POV_time_B) / 2
```

The driving-data anchor is then defined as the ASC frame whose `TimeMS` is closest to this mean POV time:

```text
ASC Anchor TimeMS
= ASC TimeMS closest to Mean Scenario_1_Start POV Time
```

---

## 2. `target_TimeMS` and `matched_TimeMS`

`target_TimeMS` is the theoretical mapped time of an eye-tracking event on the driving-data timeline after synchronization.

For all cases, the same synchronization formula is used:

```text
target_TimeMS
= ASC Anchor TimeMS
  + (Eye-tracking Time - Final Eye-Tracking Anchor Time) × 1000
```
Because ASC data are discrete sampled data at approximately 60 Hz, the theoretical `target_TimeMS` usually does not exactly match an existing driving-data frame. Therefore, the closest available ASC frame should be selected:

```text
matched_TimeMS
= ASC TimeMS closest to target_TimeMS
```

where:

- `Eye-tracking Time` is the BORIS timestamp in seconds;
- `Final Eye-Tracking Anchor Time` is the synchronization anchor on the eye-tracking annotation timeline;
- `ASC Anchor TimeMS` is the corresponding driving-data anchor on the ASC timeline, determined according to Section 1;
- `target_TimeMS` is the theoretical mapped time of the eye-tracking event on the original ASC time axis.

- `matched_TimeMS` is the actual sampled frame time in the ASC file that is closest to `target_TimeMS`.



The original `target_TimeMS` should be retained together with `matched_TimeMS` so that the matching difference can be inspected later.

---

## 3. Gaze Event

The annotated gaze behavior is:

```text
Gaze_to_initiator
```

Each gaze episode should be annotated as a state event with:

```text
START
STOP
```

Both START and STOP times are synchronized to the ASC `TimeMS` timeline using the synchronization formula defined in Section 2.

---

## 4. Summary of Synchronization Logic

| Case | Eye-tracking anchor source | Final Eye-Tracking Anchor Time | Driving-data anchor source | ASC Anchor TimeMS |
|---|---|---|---|---|
| Run Start available | Two annotators' `RunN_Start` annotation times | Mean of the two `RunN_Start` times | Corresponding ASC file | First valid ASC `TimeMS` |
| `P06 AV0` | Two annotators' `Scenario_2_Start` eye-tracking annotation times | Mean of `Scenario_2_Start_eye_time_A` and `Scenario_2_Start_eye_time_B` | Two annotators' independently identified `Scenario_2_Start` POV times | ASC `TimeMS` closest to the mean POV `Scenario_2_Start` time |
| `P21 HV1` | Two annotators' `Scenario_1_Start` eye-tracking annotation times | Mean of `Scenario_1_Start_eye_time_A` and `Scenario_1_Start_eye_time_B` | Two annotators' independently identified `Scenario_1_Start` POV times | ASC `TimeMS` closest to the mean POV `Scenario_1_Start` time |

For the two missing-`RunN_Start` cases, the eye-tracking anchor and the driving-data anchor must not be treated as the same source:

- the eye-tracking anchor is derived from the two annotators' eye-tracking annotation times;
- the driving-data anchor is derived from the two annotators' POV-based scenario-start times and then matched to the nearest ASC frame.

All original anchor measurements should be retained for traceability.
