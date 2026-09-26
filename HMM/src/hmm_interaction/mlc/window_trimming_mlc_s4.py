from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmm_interaction.io import write_csv
from mlc_s4_coordinates import build_mlc_s4_coordinates, numeric


MLC_S4_DERIVED_COLUMNS = (
    "time_sec",
    "ego_x_main",
    "ego_y_main",
    "initiator_x_main",
    "initiator_y_main",
    "initiator_y_main_smooth",
    "initiator_v_y_main",
    "initiator_a_y_main",
    "initiator_raw_lane_id",
)

MLC_S4_SUMMARY_COLUMNS = (
    "event_id",
    "condition",
    "initial_detection_group",
    "initial_failure_reason",
    "fallback_failure_reason",
    "final_output_group",
    "t_end_method",
    "warning_revised_rule",
    "event_start_TimeMS",
    "event_start_source_row",
    "lane_cross_TimeMS",
    "lane_cross_source_row",
    "lane_change_end_TimeMS",
    "lane_change_end_source_row",
    "t_cross_sec",
    "t_lane7_center_cross_sec",
    "t_end_sec",
    "n_frames_input",
    "n_frames_trimmed",
    "initiator_v_y_main_at_end",
    "initiator_a_y_main_at_end",
    "warnings",
)


# 返回由 condition/MLC/4 目录组成的正常 Scenario 4 preliminary 文件。
def find_mlc_s4_input_files(input_root: str | Path, conditions: Iterable[str]) -> list[Path]:
    root = Path(input_root)
    files: list[Path] = []
    for condition in conditions:
        files.extend(
            path
            for path in (root / condition / "MLC" / "4").glob("*_interaction.csv")
            if not path.name.startswith("._")
        )
    return sorted(files)


# 从 preliminary CSV 读取 MLC Scenario 4 必需 metadata，并验证事件类型。
def metadata_from_frame(frame: pd.DataFrame, file_path: Path) -> dict[str, Any]:
    required = ("event_id", "condition", "scenario_id", "initiator_prefix")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"MLC Scenario 4 缺少 event metadata 列：{', '.join(missing)}")
    first = frame.iloc[0]
    scenario_id = pd.to_numeric(pd.Series([first["scenario_id"]]), errors="coerce").iloc[0]
    if not np.isfinite(scenario_id) or int(scenario_id) != 4:
        raise ValueError(f"MLC trimming 仅支持 scenario_id 4，当前为 {first['scenario_id']}。")
    return {
        "event_id": str(first["event_id"]),
        "condition": str(first["condition"]),
        "scenario_id": 4,
        "initiator_prefix": str(first["initiator_prefix"]),
        "input_path": str(file_path),
    }


# 在布尔条件中返回首个实际时间长度不少于 duration 的连续区间起点。
def first_continuous_window_start(
    time_sec: np.ndarray,
    condition: np.ndarray,
    duration_sec: float,
) -> float | None:
    start: int | None = None
    for index, value in enumerate(condition):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(condition) - 1):
            end = index if value and index == len(condition) - 1 else index - 1
            if time_sec[end] - time_sec[start] >= duration_sec - 1e-9:
                return float(time_sec[start])
            start = None
    return None


# 以固定 lane 6/7 boundary 检测 initiator 首次从 lane 6 侧进入 lane 7 侧的时刻。
def detect_s4_y_crossing(frame: pd.DataFrame, boundary_y_m: float) -> float | None:
    time = numeric(frame, "time_sec").to_numpy(float)
    y_main = numeric(frame, "initiator_y_main").to_numpy(float)
    seen_lane6_side = False
    for index in range(len(frame)):
        if not np.isfinite(time[index]) or not np.isfinite(y_main[index]):
            continue
        if y_main[index] < boundary_y_m:
            seen_lane6_side = True
        if seen_lane6_side and y_main[index] >= boundary_y_m:
            return float(time[index])
    return None


# 记录 raw LaneID 明确发生 6→7 的候选时刻；三位数 LaneID 不映射也不视为失败。
def detect_s4_laneid_crossing_candidate(frame: pd.DataFrame) -> float | None:
    time = numeric(frame, "time_sec").to_numpy(float)
    lane = numeric(frame, "initiator_raw_lane_id").to_numpy(float)
    seen_lane6 = False
    for index, lane_id in enumerate(lane):
        if np.isfinite(lane_id) and np.isclose(lane_id, 6.0):
            seen_lane6 = True
        if seen_lane6 and np.isfinite(lane_id) and np.isclose(lane_id, 7.0):
            return float(time[index])
    return None


# 以 y-based crossing 为正式锚点，同时记录 LaneID 候选与坐标变化不一致的 warning。
def choose_s4_crossing_anchor(
    t_cross_y: float | None,
    t_cross_lane_id: float | None,
    warning_threshold_sec: float,
) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    if t_cross_y is None:
        warnings.append("missing_y_based_crossing")
        return None, warnings
    if t_cross_lane_id is not None and abs(t_cross_y - t_cross_lane_id) > warning_threshold_sec:
        warnings.append("crossing_y_laneid_difference_exceeds_warning_threshold")
    return t_cross_y, warnings


# 在 crossing 后的 lane 7 一侧查找横向速度、加速度均连续稳定的最早窗口。
def detect_s4_stable_target_lane(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    boundary_y_m: float,
    config: dict[str, Any],
    acceleration_threshold: float | None = None,
    t_departure_sec: float | None = None,
) -> float | None:
    if t_cross_sec is None:
        return None
    settings = config["end_detection"]
    acceleration_limit = float(
        settings["lateral_acc_threshold_mps2"]
        if acceleration_threshold is None
        else acceleration_threshold
    )
    time = numeric(frame, "time_sec").to_numpy(float)
    y_main = numeric(frame, "initiator_y_main").to_numpy(float)
    velocity = numeric(frame, "initiator_v_y_main").to_numpy(float)
    acceleration = numeric(frame, "initiator_a_y_main").to_numpy(float)
    before_departure = np.ones(len(frame), dtype=bool)
    if t_departure_sec is not None:
        before_departure = time < t_departure_sec
    stable = (
        (time >= t_cross_sec)
        & before_departure
        & np.isfinite(y_main)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & (y_main >= boundary_y_m)
        & (np.abs(velocity) < float(settings["lateral_velocity_threshold_mps"]))
        & (np.abs(acceleration) < acceleration_limit)
    )
    return first_continuous_window_start(time, stable, float(settings["stable_duration_sec"]))


# 返回 crossing 后最早接近 lane 7 center 的空间诊断候选，不参与正式终点判断。
def detect_s4_spatial_candidate(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    tolerance_m: float,
) -> float | None:
    if t_cross_sec is None:
        return None
    time = numeric(frame, "time_sec")
    y_main = numeric(frame, "initiator_y_main")
    eligible = time.ge(t_cross_sec) & y_main.abs().lt(tolerance_m)
    return float(time.loc[eligible].iloc[0]) if eligible.any() else None


# 在进入 lane 7 一侧后，以持续偏离 lane 7 center 的规则定位 off-ramp 横移起点。
def detect_s4_lane7_departure(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    config: dict[str, Any],
) -> float | None:
    if t_cross_sec is None:
        return None
    settings = config["lane7_departure"]
    time = numeric(frame, "time_sec").to_numpy(float)
    y_main = numeric(frame, "initiator_y_main").to_numpy(float)
    condition = (
        (time >= t_cross_sec)
        & np.isfinite(y_main)
        & (np.abs(y_main) > float(settings["abs_y_main_threshold_m"]))
    )
    return first_continuous_window_start(time, condition, float(settings["continuous_duration_sec"]))


# 严格主规则失败时，在首个放宽加速度但仍保持速度和持续时长的窗口内选取最接近 lane 7 center 的时刻。
def detect_s4_relaxed_kinematic_fallback(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    boundary_y_m: float,
    t_departure_sec: float | None,
    config: dict[str, Any],
) -> tuple[float | None, float | None]:
    if t_cross_sec is None:
        return None, None
    settings = config["end_detection"]
    fallback = config["fallback"]
    time = numeric(frame, "time_sec").to_numpy(float)
    y_main = numeric(frame, "initiator_y_main").to_numpy(float)
    velocity = numeric(frame, "initiator_v_y_main").to_numpy(float)
    acceleration = numeric(frame, "initiator_a_y_main").to_numpy(float)
    eligible = (
        (time >= t_cross_sec)
        & np.isfinite(y_main)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & (y_main >= boundary_y_m)
        & (np.abs(velocity) < float(settings["lateral_velocity_threshold_mps"]))
        & (np.abs(acceleration) < float(fallback["lateral_acc_threshold_mps2"]))
    )
    if t_departure_sec is not None:
        eligible &= time < t_departure_sec

    duration_sec = float(fallback["stable_duration_sec"])
    start: int | None = None
    for index, is_eligible in enumerate(eligible):
        if is_eligible and start is None:
            start = index
        if start is None or (is_eligible and index != len(eligible) - 1):
            continue
        end = index if is_eligible else index - 1

        if time[end] - time[start] >= duration_sec - 1e-9:
            
            window_end = int(np.flatnonzero(time <= time[start] + duration_sec)[-1])
            best_index = start + int(np.abs(y_main[start : window_end + 1]).argmin())
            return float(time[best_index]), float(time[start])
        start = None
    return None, None


# 在平滑横向位置中定位 t_cross 后首次由负到正穿过 lane 7 center 的插值时刻，并确认其后持续位于中心右侧。
def detect_s4_lane7_center_crossing(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    config: dict[str, Any],
) -> float | None:
    if t_cross_sec is None:
        return None
    time = numeric(frame, "time_sec").to_numpy(float)
    smooth_y = numeric(frame, "initiator_y_main_smooth").to_numpy(float)
    confirmation = float(config["warning_revised"]["center_confirmation_duration_sec"])
    for index in range(1, len(frame)):
        left_time, right_time = time[index - 1], time[index]
        left_y, right_y = smooth_y[index - 1], smooth_y[index]
        if (
            not np.isfinite(left_time)
            or not np.isfinite(right_time)
            or not np.isfinite(left_y)
            or not np.isfinite(right_y)
            or right_time < t_cross_sec
            or left_y >= 0.0
            or right_y < 0.0
            or np.isclose(right_y, left_y)
        ):
            continue
        # 在相邻 60 Hz 实际帧之间按平滑位置做线性插值，避免把 center crossing 固定到后一帧。
        center_time = left_time + (-left_y) * (right_time - left_time) / (right_y - left_y)
        after_center = (time >= center_time) & np.isfinite(smooth_y) & (smooth_y >= 0.0)
        if first_continuous_window_start(time, after_center, confirmation) is not None:
            return float(center_time)
    return None


# 在永久离开 lane 7 邻域前，选择最后一个持续低速段的最后实际帧作为 warning-revised2 终点。
def detect_s4_last_pre_offramp_low_velocity(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    config: dict[str, Any],
) -> tuple[float | None, float | None, bool]:
    if t_cross_sec is None:
        return None, None, False
    settings = config["warning_revised"]
    time = numeric(frame, "time_sec").to_numpy(float)
    y_main = numeric(frame, "initiator_y_main").to_numpy(float)
    velocity = numeric(frame, "initiator_v_y_main").to_numpy(float)
    exit_condition = (
        (time >= t_cross_sec)
        & np.isfinite(y_main)
        & (y_main > float(settings["lane7_corridor_max_y_m"]))
    )
    exit_duration = float(settings["lane7_corridor_exit_confirmation_duration_sec"])
    corridor_exit: float | None = None
    start: int | None = None
    for index, is_outside in enumerate(exit_condition):
        if is_outside and start is None:
            start = index
        if start is None or (is_outside and index != len(exit_condition) - 1):
            continue
        end = index if is_outside else index - 1
        # 短暂驶到 y>0.5 m 后又回到 lane 7 邻域并不表示驶入 off-ramp；仅无再进入时才视为永久离开。
        if time[end] - time[start] >= exit_duration - 1e-9 and exit_condition[end:].all():
            corridor_exit = float(time[start])
            break
        start = None
    eligible = (
        (time >= t_cross_sec)
        & np.isfinite(velocity)
        & (np.abs(velocity) < float(settings["lateral_velocity_threshold_mps"]))
    )
    if corridor_exit is not None:
        eligible &= time < corridor_exit

    duration = float(settings["stable_duration_sec"])
    last_end: int | None = None
    start = None
    for index, is_eligible in enumerate(eligible):
        if is_eligible and start is None:
            start = index
        if start is None or (is_eligible and index != len(eligible) - 1):
            continue
        end = index if is_eligible else index - 1
        if time[end] - time[start] >= duration - 1e-9:
            # 保留最后一个合格段的最后实际帧，即随后重新横向加速前的最后低速帧。
            last_end = end
        start = None

    late_low_velocity_segment_excluded = False
    if corridor_exit is not None:
        late_eligible = (
            (time >= corridor_exit)
            & np.isfinite(velocity)
            & (np.abs(velocity) < float(settings["lateral_velocity_threshold_mps"]))
        )
        late_low_velocity_segment_excluded = (
            first_continuous_window_start(time, late_eligible, duration) is not None
        )
    return (
        float(time[last_end]) if last_end is not None else None,
        corridor_exit,
        late_low_velocity_segment_excluded,
    )


# 对 primary 与普通 fallback 都失败的 event，先使用最后低速段；无候选时才回退到 center crossing。
def detect_s4_warning_revised_end(
    frame: pd.DataFrame,
    t_cross_sec: float | None,
    config: dict[str, Any],
) -> tuple[float | None, str | None, float | None, bool, float | None, bool]:
    t_center = detect_s4_lane7_center_crossing(frame, t_cross_sec, config)
    low_velocity_end, corridor_exit, late_excluded = detect_s4_last_pre_offramp_low_velocity(
        frame, t_cross_sec, config
    )
    if low_velocity_end is not None:
        return low_velocity_end, "last_pre_offramp_low_velocity", t_center, False, corridor_exit, late_excluded
    if t_center is not None:
        # center crossing 是插值时刻；裁剪时必须保留其后的首个真实 60 Hz 帧。
        return t_center, "direct_offramp_center_crossing", t_center, True, corridor_exit, late_excluded
    return None, None, None, False, corridor_exit, late_excluded


# 保留至 end 时刻及可选缓冲；没有 end 时保留完整已构造坐标的窗口。
def trim_s4_interaction_window(
    frame: pd.DataFrame,
    t_end_sec: float | None,
    config: dict[str, Any],
    include_first_frame_after_end: bool = False,
) -> pd.DataFrame:
    if t_end_sec is None:
        return frame.copy()
    if include_first_frame_after_end:
        indices = np.flatnonzero(numeric(frame, "time_sec").to_numpy(float) >= t_end_sec)
        return frame.iloc[: int(indices[0]) + 1].copy() if len(indices) else frame.copy()
    buffer_sec = float(config["end_detection"].get("post_end_buffer_sec", 0.0))
    return frame.loc[numeric(frame, "time_sec").le(t_end_sec + buffer_sec)].copy()


# 从 event-local 时刻回查不晚于该时刻的原始 TimeMS 与 source_row。
def time_and_source_row_at_or_before(
    frame: pd.DataFrame,
    time_sec: float | None,
) -> tuple[float | None, int | None]:
    if time_sec is None or "source_row" not in frame:
        return None, None
    candidates = frame.loc[numeric(frame, "time_sec").le(time_sec)]
    if candidates.empty:
        return None, None
    row = candidates.iloc[-1]
    time_ms = pd.to_numeric(pd.Series([row["TimeMS"]]), errors="coerce").iloc[0]
    source_row = pd.to_numeric(pd.Series([row["source_row"]]), errors="coerce").iloc[0]
    return (
        float(time_ms) if np.isfinite(time_ms) else None,
        int(source_row) if np.isfinite(source_row) else None,
    )


# 从 event-local 插值时刻回查不早于该时刻的首个实际原始帧。
def time_and_source_row_at_or_after(
    frame: pd.DataFrame,
    time_sec: float | None,
) -> tuple[float | None, int | None]:
    if time_sec is None or "source_row" not in frame:
        return None, None
    candidates = frame.loc[numeric(frame, "time_sec").ge(time_sec)]
    if candidates.empty:
        return None, None
    row = candidates.iloc[0]
    time_ms = pd.to_numeric(pd.Series([row["TimeMS"]]), errors="coerce").iloc[0]
    source_row = pd.to_numeric(pd.Series([row["source_row"]]), errors="coerce").iloc[0]
    return (
        float(time_ms) if np.isfinite(time_ms) else None,
        int(source_row) if np.isfinite(source_row) else None,
    )


# 将 event 起点、crossing 与正式终点对应的原始时刻和 source_row 写入 summary。
def add_event_timing(
    record: dict[str, Any],
    frame: pd.DataFrame,
    t_cross: float | None,
    t_end: float | None,
    end_is_interpolated: bool = False,
) -> None:
    start_time, start_row = time_and_source_row_at_or_before(frame, 0.0)
    cross_time, cross_row = time_and_source_row_at_or_before(frame, t_cross)
    end_time, end_row = (
        time_and_source_row_at_or_after(frame, t_end)
        if end_is_interpolated
        else time_and_source_row_at_or_before(frame, t_end)
    )
    record.update(
        {
            "event_start_TimeMS": start_time,
            "event_start_source_row": start_row,
            "lane_cross_TimeMS": cross_time,
            "lane_cross_source_row": cross_row,
            "lane_change_end_TimeMS": end_time,
            "lane_change_end_source_row": end_row,
        }
    )


# 返回起始 0.5 秒 raw LaneID 的稳定众数，用于 summary 的输入审计。
def initial_raw_lane_id(frame: pd.DataFrame) -> float | None:
    values = pd.to_numeric(
        frame.loc[numeric(frame, "time_sec").le(0.5), "initiator_raw_lane_id"],
        errors="coerce",
    ).dropna()
    return float(values.mode().min()) if not values.empty else None


# 绘制 Scenario 4 的固定 shared-coordinate、运动学和 raw LaneID 诊断图。
def create_mlc_s4_diagnostic_plot(
    frame: pd.DataFrame,
    record: dict[str, Any],
    output_path: str | Path,
    config: dict[str, Any],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    limits = config["diagnostics"]["axis_limits"]
    coordinate = config["coordinate"]
    time = numeric(frame, "time_sec")
    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(time, numeric(frame, "initiator_y_main"), alpha=0.45, label="y_main")
    axes[0].plot(time, numeric(frame, "initiator_y_main_smooth"), label="y_main smoothed")
    axes[0].axhline(float(coordinate["initial_initiator_lane_center_y_m"]), color="tab:gray", linestyle="--", label="lane 6 center")
    axes[0].axhline(float(coordinate["lane6_lane7_boundary_y_m"]), color="black", linestyle="--", label="lane 6/7 boundary")
    axes[0].axhline(float(coordinate["target_lane_center_y_m"]), color="tab:green", linestyle="-.", label="lane 7 center")
    axes[0].set_ylim(limits["y_main_m"])
    axes[0].set_ylabel("shared lateral position [m]")
    axes[0].legend(loc="best")

    velocity_limit = float(record["v_y_threshold_mps"])
    axes[1].plot(time, numeric(frame, "initiator_v_y_main"), label="v_y_main")
    axes[1].axhline(velocity_limit, color="tab:purple", linestyle="--", label=f"{velocity_limit:.2f} velocity threshold")
    axes[1].axhline(-velocity_limit, color="tab:purple", linestyle="--")
    axes[1].set_ylim(limits["v_y_mps"])
    axes[1].set_ylabel("lateral velocity [m/s]")
    axes[1].legend(loc="best")

    acceleration_limit = float(record["a_y_threshold_mps2"])
    sensitivity_limit = float(record["a_y_sensitivity_threshold_mps2"])
    axes[2].plot(time, numeric(frame, "initiator_a_y_main"), label="a_y_main")
    axes[2].axhline(acceleration_limit, color="tab:red", linestyle="--", label=f"{acceleration_limit:.2f} primary threshold")
    axes[2].axhline(-acceleration_limit, color="tab:red", linestyle="--")
    axes[2].axhline(sensitivity_limit, color="tab:orange", linestyle=":", label=f"{sensitivity_limit:.2f} sensitivity")
    axes[2].axhline(-sensitivity_limit, color="tab:orange", linestyle=":")
    fallback_acceleration_limit = record.get("fallback_a_y_threshold_mps2")
    if record.get("end_detection_status") == "fallback" and fallback_acceleration_limit is not None:
        axes[2].axhline(
            float(fallback_acceleration_limit),
            color="tab:gray",
            linestyle="-.",
            label=f"{float(fallback_acceleration_limit):.2f} fallback threshold",
        )
        axes[2].axhline(-float(fallback_acceleration_limit), color="tab:gray", linestyle="-.")
    axes[2].set_ylim(limits["a_y_mps2"])
    axes[2].set_ylabel("lateral acceleration [m/s²]")
    axes[2].legend(loc="best")

    raw_lane = numeric(frame, "initiator_raw_lane_id")
    axes[3].step(time, raw_lane, where="post", label="raw LaneID")
    finite_lane = raw_lane[np.isfinite(raw_lane)]
    if finite_lane.empty:
        axes[3].set_ylim(4.5, 7.5)
    else:
        lower = min(4.5, float(finite_lane.min()) - 1.0)
        upper = max(7.5, float(finite_lane.max()) + 1.0)
        axes[3].set_ylim(lower, upper)
    for lane_id in (5.0, 6.0, 7.0):
        axes[3].axhline(lane_id, color="tab:gray", linestyle=":", linewidth=0.8)
    axes[3].set_ylabel("raw LaneID")
    axes[3].set_xlabel("event-local time [s]")
    axes[3].legend(loc="best")

    # 图中仅保留橙色 final t_end 竖线；其余检测时刻在 summary 中追溯，避免多条竖线混淆人工判读。
    markers = (("t_end_sec", "tab:orange", "final t_end"),)
    for axis in axes:
        for key, color, label in markers:
            value = record.get(key)
            if value is not None and np.isfinite(value):
                axis.axvline(
                    float(value),
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                    label=label if axis is axes[0] else None,
                )
    final_end = record.get("t_end_sec")
    final_text = "final t_end=not detected" if final_end is None or not np.isfinite(final_end) else f"final t_end={float(final_end):.3f} s"
    axes[0].legend(loc="best")
    figure.suptitle(f"{record['event_id']} | {record['t_end_method']} | {final_text}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# 计算单个 Scenario 4 window 的终点和分类；坐标有效但仍未能检测终点时不在此处写出 CSV。
def evaluate_mlc_s4_window(
    file_path: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bool]:
    input_path = Path(file_path)
    original = pd.read_csv(input_path)
    if original.empty:
        raise ValueError("无法裁剪空的 MLC Scenario 4 preliminary window。")
    metadata = metadata_from_frame(original, input_path)
    frame, _ = build_mlc_s4_coordinates(original, metadata, config)
    coordinate = config["coordinate"]
    crossing = config["crossing"]
    settings = config["end_detection"]
    boundary = float(coordinate["lane6_lane7_boundary_y_m"])
    warnings: list[str] = []
    t_cross_y = detect_s4_y_crossing(frame, boundary)
    t_cross_lane = detect_s4_laneid_crossing_candidate(frame)
    t_cross, crossing_warnings = choose_s4_crossing_anchor(
        t_cross_y,
        t_cross_lane,
        float(crossing["crossing_time_diff_warning_sec"]),
    )
    warnings.extend(crossing_warnings)
    t_departure = detect_s4_lane7_departure(frame, t_cross, config)
    t_stable = detect_s4_stable_target_lane(frame, t_cross, boundary, config, t_departure_sec=t_departure)
    t_sensitivity = detect_s4_stable_target_lane(
        frame,
        t_cross,
        boundary,
        config,
        float(settings["sensitivity_lateral_acc_threshold_mps2"]),
        t_departure,
    )

    t_end: float | None = None
    t_fallback_stable: float | None = None
    method = "keep_original_window"
    status = "warning"
    initial_group = "warning"
    initial_failure_reason: str | None = None
    fallback_failure_reason: str | None = None
    warning_rule: str | None = None
    end_is_interpolated = False
    t_center_cross: float | None = detect_s4_lane7_center_crossing(frame, t_cross, config)
    t_warning_revised_corridor_exit: float | None = None
    warning_revised_late_low_velocity_segment_excluded = False
    if t_cross is not None and t_stable is not None:
        t_end, method, status = t_stable, "stable_target_lane", "success"
        initial_group = "success"
    elif t_cross is not None:
        initial_failure_reason = "missing_strict_stable_target_lane_window"
        t_end, t_fallback_stable = detect_s4_relaxed_kinematic_fallback(
            frame,
            t_cross,
            boundary,
            t_departure,
            config,
        )
        if t_end is None:
            warnings.append("missing_relaxed_kinematic_fallback_window")
            fallback_failure_reason = "missing_relaxed_kinematic_fallback_window"
            (
                t_end,
                warning_rule,
                t_center_cross,
                end_is_interpolated,
                t_warning_revised_corridor_exit,
                warning_revised_late_low_velocity_segment_excluded,
            ) = detect_s4_warning_revised_end(
                frame, t_cross, config
            )
            if t_end is not None:
                method = warning_rule
                status = "warning_revised2"
        else:
            method, status = "lane7_relaxed_kinematic_alignment", "fallback"
            initial_group = "fallback"
            if t_departure is None:
                warnings.append("missing_lane7_departure_fallback_searched_to_window_end")
    elif t_cross is None:
        initial_failure_reason = "missing_y_based_crossing"

    trimmed = trim_s4_interaction_window(frame, t_end, config, end_is_interpolated)
    trimmed = trimmed.drop(columns=["window_rule", "scenario_segment_id", "source_file", "segmentation_rule"], errors="ignore")
    crossing_difference = abs(t_cross_y - t_cross_lane) if t_cross_y is not None and t_cross_lane is not None else None
    record: dict[str, Any] = {
        "event_id": metadata["event_id"],
        "condition": metadata["condition"],
        "scenario_id": 4,
        "input_path": str(input_path),
        "n_frames_input": len(original),
        "n_frames_trimmed": len(trimmed),
        "initiator_initial_raw_lane_id": initial_raw_lane_id(frame),
        "t_cross_y_sec": t_cross_y,
        "t_cross_lane_id_sec": t_cross_lane,
        "t_cross_sec": t_cross,
        "crossing_time_difference_sec": crossing_difference,
        "t_stable_start_sec": t_stable,
        "t_fallback_stable_start_sec": t_fallback_stable,
        "t_lane7_departure_sec": t_departure,
        "t_lane7_center_cross_sec": t_center_cross,
        "t_end_sec": t_end,
        "t_end_sensitivity_005_sec": t_sensitivity,
        "t_end_method": method,
        "end_detection_status": status,
        "initial_detection_group": initial_group,
        "initial_failure_reason": initial_failure_reason,
        "fallback_failure_reason": fallback_failure_reason,
        "final_output_group": status if status in {"success", "fallback", "warning_revised2"} else None,
        "warning_revised_rule": warning_rule,
        "warning_revised_corridor_exit_sec": t_warning_revised_corridor_exit,
        "warning_revised_late_low_velocity_segment_excluded": warning_revised_late_low_velocity_segment_excluded,
        "v_y_threshold_mps": float(settings["lateral_velocity_threshold_mps"]),
        "a_y_threshold_mps2": float(settings["lateral_acc_threshold_mps2"]),
        "a_y_sensitivity_threshold_mps2": float(settings["sensitivity_lateral_acc_threshold_mps2"]),
        "fallback_a_y_threshold_mps2": float(config["fallback"]["lateral_acc_threshold_mps2"]),
    }
    for tolerance in config["spatial_diagnostics"]["target_center_tolerances_m"]:
        key = int(round(float(tolerance) * 100))
        record[f"t_end_spatial_{key:03d}_sec"] = detect_s4_spatial_candidate(frame, t_cross, float(tolerance))
    if t_end is not None:
        end_rows = frame.loc[
            numeric(frame, "time_sec").ge(t_end)
            if end_is_interpolated
            else numeric(frame, "time_sec").le(t_end)
        ]
        end_index = end_rows.index[0] if end_is_interpolated else end_rows.index[-1]
        record.update(
            {
                "initiator_y_main_at_end": float(frame.loc[end_index, "initiator_y_main"]),
                "initiator_v_y_main_at_end": float(frame.loc[end_index, "initiator_v_y_main"]),
                "initiator_a_y_main_at_end": float(frame.loc[end_index, "initiator_a_y_main"]),
            }
        )
    add_event_timing(record, frame, t_cross, t_end, end_is_interpolated)
    record["warnings"] = "; ".join(warnings)
    return frame, trimmed, record, end_is_interpolated


# 处理单个 Scenario 4 preliminary window，并保持旧的单文件调用方式供单元测试和局部复核使用。
def process_one_mlc_s4_window(
    file_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    frame, trimmed, record, _ = evaluate_mlc_s4_window(file_path, config)
    write_csv(trimmed, output_path)
    if diagnostics_path is not None:
        create_mlc_s4_diagnostic_plot(frame, record, diagnostics_path, config)
    return record


# 创建不覆盖已有 pilot 或正式诊断图的时间戳目录。
def create_unique_output_root(root: str | Path, label: str) -> Path:
    base = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{timestamp}_{label}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{timestamp}_{label}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


# 为未能构造坐标的 event 生成失败记录，且不写出不可信的 trimmed CSV。
def failed_record(file_path: Path, error: Exception) -> dict[str, Any]:
    return {
        "event_id": file_path.stem.removesuffix("_interaction"),
        "condition": None,
        "scenario_id": 4,
        "n_frames_input": None,
        "n_frames_trimmed": None,
        "initial_detection_group": "warning",
        "initial_failure_reason": "invalid_coordinate_input",
        "fallback_failure_reason": None,
        "final_output_group": None,
        "warning_revised_rule": None,
        "t_end_method": "not_written_invalid_coordinate_input",
        "end_detection_status": "failed",
        "warnings": str(error),
    }


# 批量执行 pilot、全量 test 或确认后的正式 Scenario 4 trimming，并隔离各类输出。
def run_mlc_s4_window_trimming(
    config: dict[str, Any],
    subject_id: str | None = None,
    diagnostics: bool = False,
    run_metadata: dict[str, Any] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    files = find_mlc_s4_input_files(config["input"]["root"], config["input"]["conditions"])
    if subject_id is not None:
        files = [path for path in files if path.stem.startswith(f"{subject_id}_")]
    if not files:
        raise FileNotFoundError("未找到符合条件的 MLC Scenario 4 preliminary window。")

    output = config["output"]
    label = "all" if subject_id is None else subject_id
    run_root = create_unique_output_root(Path(output["test_root"]), label) if (test_mode or subject_id is not None) else None
    if run_root is not None:
        trimmed_root = run_root / "trimmed_windows"
        summary_path = run_root / "mlc_s4_end_detection_test_summary.csv"
        diagnostic_root = run_root / "diagnostics" if diagnostics else None
    else:
        trimmed_root = Path(output["trimmed_root"])
        summary_path = trimmed_root / "mlc_s4_end_detection_summary.csv"
        diagnostic_root = create_unique_output_root(Path(output["diagnostics_root"]), "all") if diagnostics else None

    records: list[dict[str, Any]] = []
    for path in files:
        try:
            frame, trimmed, record, _ = evaluate_mlc_s4_window(path, config)
            output_group = record["final_output_group"]
            if output_group is not None:
                output_path = trimmed_root / output_group / f"{path.stem}_trimmed.csv"
                write_csv(trimmed, output_path)
                if diagnostic_root is not None:
                    diagnostic_path = diagnostic_root / output_group / f"{path.stem}_end_detection.png"
                    create_mlc_s4_diagnostic_plot(frame, record, diagnostic_path, config)
            records.append(record)
        except (ValueError, KeyError) as error:
            records.append(failed_record(path, error))

    full_summary = pd.DataFrame(records)
    summary = full_summary.reindex(columns=MLC_S4_SUMMARY_COLUMNS)
    write_csv(summary, summary_path)
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_input_files": len(files),
        "summary_path": str(summary_path),
        "diagnostics": diagnostics,
        **(run_metadata or {}),
    }
    if run_root is not None:
        metadata["run_directory"] = str(run_root)
        (run_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    elif diagnostic_root is not None:
        metadata["diagnostics_directory"] = str(diagnostic_root)
        (diagnostic_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    statuses = full_summary.get("end_detection_status", pd.Series(dtype=object))
    return {
        "n_input_files": len(files),
        "n_success": int(statuses.eq("success").sum()),
        "n_warning": int(statuses.eq("warning").sum()),
        "n_warning_revised": int(statuses.eq("warning_revised2").sum()),
        "n_fallback": int(statuses.eq("fallback").sum()),
        "n_failed": int(statuses.eq("failed").sum()),
        "summary_path": str(summary_path),
        "run_directory": str(run_root) if run_root is not None else None,
        "diagnostics_directory": str(diagnostic_root) if diagnostic_root is not None else None,
    }
