"""DLC 变道完成时刻检测与 preliminary window 裁剪。"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

# 批量生成诊断图时使用无界面后端，避免 macOS 图形通知中断长时间批处理。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmm_interaction.dlc_coordinates import (
    EGO_LANE_COLUMN,
    build_temporary_dlc_coordinates_for_trimming,
)
from hmm_interaction.io import write_csv


DEFAULT_DIAGNOSTIC_AXIS_LIMITS: dict[str, tuple[float, float]] = {
    "y_relative_lane_boundary_m": (-3.5, 3.5),
    "v_y_mps": (-2.5, 2.5),
    "a_y_mps2": (-6.0, 6.0),
}
DEFAULT_LANE_ID_AXIS_LIMITS: dict[int, tuple[float, float]] = {
    2: (5.5, 7.5),
    3: (4.5, 6.5),
}
TRIMMED_REDUNDANT_METADATA_COLUMNS = (
    "window_id",
    "window_rule",
    "source_file",
    "segmentation_rule",
    "scenario_segment_id",
    "subject_id",
    "condition",
    "scenario_id",
    "scenario_name",
    "event_id",
    "initiator_prefix",
    "sampling_rate_hz",
)
TRIMMED_DERIVED_COORDINATE_COLUMNS = (
    "time_sec",
    "ego_x_main",
    "ego_y_main",
    "initiator_x_main",
    "initiator_y_main",
    "initiator_y_main_smooth",
    "initiator_v_y_main",
    "initiator_a_y_main",
)
COMPACT_SUMMARY_COLUMNS = (
    "event_id",
    "initial_lane_id",
    "expected_target_lane_id",
    "event_start_TimeMS",
    "event_start_source_row",
    "lane_cross_TimeMS",
    "lane_cross_source_row",
    "lane_change_end_TimeMS",
    "lane_change_end_source_row",
    "event_duration_sec",
    "n_frames_input",
    "n_frames_trimmed",
    "initiator_v_y_at_end",
    "initiator_a_y_at_end",
    "end_detection_status",
    "end_method",
    "warnings",
)


def _numeric(values: pd.Series) -> pd.Series:
    """将序列转换为数值，并将无效值保留为 NaN。"""
    return pd.to_numeric(values, errors="coerce")


def _join_notes(notes: Iterable[str]) -> str:
    """将非空诊断信息合并为 summary 的 warnings 字段。"""
    return "; ".join(note for note in notes if note)


def _axis_limit_pair(values: Any, key: str) -> tuple[float, float]:
    """校验并返回一个诊断图纵轴范围。
    """
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"诊断图坐标范围 {key} 必须包含两个数值。")
    lower, upper = float(values[0]), float(values[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"诊断图坐标范围 {key} 必须满足有限下限小于上限。")
    return lower, upper


def _diagnostic_axis_limits(
    config: dict[str, Any], scenario_id: int
) -> dict[str, tuple[float, float]]:
    """返回指定 DLC scenario 的诊断图纵轴范围。
    """
    configured = config.get("diagnostics", {}).get("axis_limits", {})
    limits: dict[str, tuple[float, float]] = {}
    for key, default in DEFAULT_DIAGNOSTIC_AXIS_LIMITS.items():
        limits[key] = _axis_limit_pair(configured.get(key, default), key)
    configured_lane_limits = configured.get("lane_id_by_scenario", {})
    default_lane_limits = DEFAULT_LANE_ID_AXIS_LIMITS.get(scenario_id)
    if default_lane_limits is None:
        raise ValueError(f"DLC 诊断图仅支持 scenario 2 或 3，当前为 {scenario_id}。")
    values = configured_lane_limits.get(
        str(scenario_id), configured_lane_limits.get(scenario_id, default_lane_limits)
    )
    limits["lane_id"] = _axis_limit_pair(values, f"lane_id_by_scenario.{scenario_id}")
    return limits


def _diagnostic_y_display(values: pd.Series | np.ndarray, lane_boundary_y_m: float) -> np.ndarray:
    """将横向位置转换为以实际 lane boundary 为零点的绘图坐标。
    """
    return np.asarray(values, dtype=float) - lane_boundary_y_m


def fixed_lane_geometry_display_reference(
    scenario_id: int,
    lane_width_m: float,
) -> tuple[float, float]:
    """返回固定车道几何下的目标与初始车道中心显示坐标，单位为米。
    """
    half_width = float(lane_width_m) / 2.0
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("诊断图车道宽度必须为正的有限数值。")
    if scenario_id == 2:
        return -half_width, half_width
    if scenario_id == 3:
        return half_width, -half_width
    raise ValueError(f"固定车道几何仅支持 scenario 2 或 3，当前为 {scenario_id}。")


def _mark_clipped_values(
    axis: Any,
    values: np.ndarray,
    limits: tuple[float, float],
) -> None:
    """当绘制数据超出固定显示范围时，在子图中标注裁切提示。"""
    finite_values = values[np.isfinite(values)]
    if finite_values.size and (finite_values.min() < limits[0] or finite_values.max() > limits[1]):
        axis.text(
            0.01,
            0.96,
            "values outside view",
            transform=axis.transAxes,
            va="top",
            color="tab:red",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "tab:red"},
        )


def _duration_sec(frame: pd.DataFrame) -> float | None:
    """根据 TimeMS 返回帧段时长（秒）；不可用时返回 ``None``。"""
    if "TimeMS" not in frame or frame.empty:
        return None
    time = _numeric(frame["TimeMS"]).dropna()
    return float((time.iloc[-1] - time.iloc[0]) / 1000.0) if len(time) else None


def find_dlc_input_files(
    input_root: str | Path,
    conditions: Iterable[str],
    scenario_ids: Iterable[int],
) -> list[Path]:
    """在 ``condition/DLC/scenario_id`` 下查找正常 DLC CSV 文件。"""
    root = Path(input_root)
    files: list[Path] = []
    for condition in conditions:
        for scenario_id in scenario_ids:
            files.extend(
                path
                for path in (root / condition / "DLC" / str(scenario_id)).glob("*.csv")
                if not path.name.startswith("._")
            )
    return sorted(files)


def infer_expected_target_lane(initial_lane_id: float, scenario_id: int) -> float:
    """推断 DLC 目标车道：scenario 2 向左，scenario 3 向右。"""
    if scenario_id == 2:
        return initial_lane_id - 1
    if scenario_id == 3:
        return initial_lane_id + 1
    raise ValueError(f"DLC trimming 仅支持 scenario_id 2 或 3，当前为 {scenario_id}。")


def _mode_in_initial_window(
    frame: pd.DataFrame,
    column: str,
    window_sec: float,
) -> float | None:
    """返回 event 起始时间窗口内确定性的数值众数。"""
    if column not in frame or "time_sec" not in frame:
        return None
    values = _numeric(frame.loc[frame["time_sec"].le(window_sec), column]).dropna()
    if values.empty:
        return None
    modes = values.mode()
    return float(modes.min())


def _mode_in_final_window(frame: pd.DataFrame, column: str, window_sec: float) -> float | None:
    """返回 event 末尾时间窗口内的数值众数。"""
    if column not in frame or "time_sec" not in frame:
        return None
    last_time = float(frame["time_sec"].iloc[-1])
    values = _numeric(frame.loc[frame["time_sec"].ge(last_time - window_sec), column]).dropna()
    if values.empty:
        return None
    return float(values.mode().min())


def detect_t_cross_by_lane_id(
    frame: pd.DataFrame,
    initial_lane_id: float,
    lane_column: str,
) -> float | None:
    """返回 initiator LaneID 首次变化的 event-local 秒数。"""
    if lane_column not in frame:
        return None
    lane = _numeric(frame[lane_column])
    changed = lane.notna() & ~np.isclose(lane, initial_lane_id)
    candidates = frame.loc[changed, "time_sec"]
    return float(candidates.iloc[0]) if not candidates.empty else None


def detect_t_cross_by_y_main(
    frame: pd.DataFrame,
    lane_boundary_y: float,
) -> float | None:
    """返回首次跨越初始/目标车道边界的 event-local 秒数。"""
    y = _numeric(frame["initiator_y_main_smooth"])
    if lane_boundary_y > 0:
        crossed = y.le(lane_boundary_y)
    elif lane_boundary_y < 0:
        crossed = y.ge(lane_boundary_y)
    else:
        return None
    candidates = frame.loc[crossed & y.notna(), "time_sec"]
    return float(candidates.iloc[0]) if not candidates.empty else None


def choose_crossing_anchor(
    t_cross_lane_id: float | None,
    t_cross_y: float | None,
    config: dict[str, Any],
) -> tuple[float | None, list[str]]:
    """选择 y-based crossing 作为锚点，并返回不一致时的警告。"""
    warnings: list[str] = []
    warning_threshold = float(
        config.get("end_detection", {}).get("crossing_time_diff_warning_sec", 0.5)
    )
    if t_cross_y is not None and t_cross_lane_id is not None:
        if abs(t_cross_y - t_cross_lane_id) > warning_threshold:
            warnings.append("crossing_time_difference_exceeds_threshold")
        return t_cross_y, warnings
    if t_cross_y is not None:
        warnings.append("missing_lane_id_crossing_using_y_crossing")
        return t_cross_y, warnings
    if t_cross_lane_id is not None:
        warnings.append("missing_y_crossing_using_lane_id_crossing")
        return t_cross_lane_id, warnings
    warnings.append("missing_crossing")
    return None, warnings


def detect_stable_completion_after_crossing(
    frame: pd.DataFrame,
    t_cross: float | None,
    config: dict[str, Any],
    acceleration_threshold: float | None = None,
    target_center_y_m: float = 0.0,
) -> float | None:
    """在 crossing 后查找首个稳定目标车道区间。
    """
    if t_cross is None:
        return None
    settings = config.get("end_detection", {})
    tolerance = float(settings.get("target_center_tolerance_m", 0.3))
    threshold = float(
        acceleration_threshold
        if acceleration_threshold is not None
        else settings.get("lateral_acc_threshold_mps2", 0.10)
    )
    velocity_threshold = float(settings.get("lateral_velocity_threshold_mps", 0.10))
    duration = float(settings.get("stable_duration_sec", 0.5))
    time = _numeric(frame["time_sec"]).to_numpy(float)
    y = _numeric(frame["initiator_y_main_smooth"]).to_numpy(float)
    velocity = _numeric(frame["initiator_v_y_main"]).to_numpy(float)
    acceleration = _numeric(frame["initiator_a_y_main"]).to_numpy(float)
    stable = (
        (time >= t_cross)
        & np.isfinite(y)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & (np.abs(y - target_center_y_m) < tolerance)
        & (np.abs(velocity) < velocity_threshold)
        & (np.abs(acceleration) < threshold)
    )

    start: int | None = None
    for index, is_stable in enumerate(stable):
        if is_stable and start is None:
            start = index
        if start is not None and (not is_stable or index == len(stable) - 1):
            end = index if is_stable and index == len(stable) - 1 else index - 1
            if time[end] - time[start] >= duration:
                return float(time[start])
            start = None
    return None


def detect_lane_kinematic_completion(
    frame: pd.DataFrame,
    target_lane_start_sec: float | None,
    expected_target_lane: float,
    lane_column: str,
    config: dict[str, Any],
    acceleration_threshold: float | None = None,
) -> float | None:
    """检测期望 LaneID 内首个横向运动稳定区间。

    ``target_lane_start_sec`` 为 initiator 首次进入期望 LaneID 的 event-local
    时刻，单位为秒。返回值为 LaneID 持续等于 ``expected_target_lane``、横向
    速度及横向加速度连续满足阈值 ``stable_duration_sec`` 的最早区间起点。
    不使用固定 ``y_main`` 目标中心，以避免道路投影漂移使结束点偏晚；缺失
    LaneID、时间或稳定区间时返回 ``None``，调用方应保留原始窗口。
    """
    if target_lane_start_sec is None or lane_column not in frame or frame.empty:
        return None
    settings = config.get("end_detection", {})
    velocity_threshold = float(settings.get("lateral_velocity_threshold_mps", 0.10))
    threshold = float(
        acceleration_threshold
        if acceleration_threshold is not None
        else settings.get("lateral_acc_threshold_mps2", 0.10)
    )
    duration = float(settings.get("stable_duration_sec", 0.5))
    time = _numeric(frame["time_sec"]).to_numpy(float)
    lane = _numeric(frame[lane_column]).to_numpy(float)
    velocity = _numeric(frame["initiator_v_y_main"]).to_numpy(float)
    acceleration = _numeric(frame["initiator_a_y_main"]).to_numpy(float)
    stable = (
        (time >= target_lane_start_sec)
        & np.isfinite(lane)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & np.isclose(lane, expected_target_lane)
        & (np.abs(velocity) < velocity_threshold)
        & (np.abs(acceleration) < threshold)
    )

    start: int | None = None
    for index, is_stable in enumerate(stable):
        if is_stable and start is None:
            start = index
        if start is not None and (not is_stable or index == len(stable) - 1):
            end = index if is_stable and index == len(stable) - 1 else index - 1
            if time[end] - time[start] >= duration:
                return float(time[start])
            start = None
    return None


def trim_window_to_end(
    frame: pd.DataFrame,
    t_end: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """保留至 ``t_end``（含配置的 post-end buffer）之前的全部帧。"""
    buffer_sec = float(config.get("end_detection", {}).get("post_end_buffer_sec", 0.0))
    return frame.loc[_numeric(frame["time_sec"]).le(t_end + buffer_sec)].copy()


def _metadata_from_frame(frame: pd.DataFrame, file_path: Path) -> dict[str, Any]:
    """从 preliminary-window CSV 读取必需的 event metadata。"""
    required = ("event_id", "condition", "scenario_id", "initiator_prefix")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少 event metadata 列：{', '.join(missing)}")
    first = frame.iloc[0]
    return {
        "event_id": str(first["event_id"]),
        "condition": str(first["condition"]),
        "scenario_id": int(first["scenario_id"]),
        "initiator_prefix": str(first["initiator_prefix"]),
        "input_path": str(file_path),
    }


def _add_trimmed_coordinate_columns(
    frame: pd.DataFrame,
    coordinate_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """写入逐帧局部坐标特征，并删除单个文件内重复的 event metadata。
    """
    result = frame.copy()
    # 这些列在单个 trimmed CSV 内恒定，且可由文件路径或 event_id 重新取得；
    # source_row 与 TimeMS 是逐帧溯源信息，必须保留。
    result = result.drop(columns=list(TRIMMED_REDUNDANT_METADATA_COLUMNS), errors="ignore")
    if coordinate_frame is None:
        time_ms = _numeric(result["TimeMS"])
        result["time_sec"] = (time_ms - time_ms.iloc[0]) / 1000.0
        for column in TRIMMED_DERIVED_COORDINATE_COLUMNS[1:]:
            result[column] = np.nan
        return result

    for column in TRIMMED_DERIVED_COORDINATE_COLUMNS:
        result[column] = coordinate_frame.loc[result.index, column]
    return result


def _time_and_source_row_at_or_before(
    coordinate_frame: pd.DataFrame | None,
    time_sec: float | None,
) -> tuple[float | None, int | None]:
    """返回不晚于 event-local 时刻的 TimeMS 与 source_row。
    """
    if coordinate_frame is None or time_sec is None or "source_row" not in coordinate_frame:
        return None, None
    candidates = coordinate_frame.loc[_numeric(coordinate_frame["time_sec"]).le(time_sec)]
    if candidates.empty:
        return None, None
    row = candidates.iloc[-1]
    time_ms = _numeric(pd.Series([row["TimeMS"]])).iloc[0]
    source_row = _numeric(pd.Series([row["source_row"]])).iloc[0]
    return (
        float(time_ms) if np.isfinite(time_ms) else None,
        int(source_row) if np.isfinite(source_row) else None,
    )


def _add_event_timing_to_record(
    record: dict[str, Any],
    original: pd.DataFrame,
    coordinate_frame: pd.DataFrame | None,
    t_cross: float | None,
    t_end: float | None,
) -> None:
    """将事件起点、crossing 与正式终点的 TimeMS/source_row 写入完整 summary。
    """
    first = original.iloc[0]
    start_time_ms = _numeric(pd.Series([first["TimeMS"]])).iloc[0]
    start_source_row = _numeric(pd.Series([first["source_row"]])).iloc[0]
    cross_time_ms, cross_source_row = _time_and_source_row_at_or_before(
        coordinate_frame, t_cross
    )
    end_time_ms, end_source_row = _time_and_source_row_at_or_before(coordinate_frame, t_end)
    record.update(
        {
            "event_start_TimeMS": float(start_time_ms) if np.isfinite(start_time_ms) else None,
            "event_start_source_row": int(start_source_row)
            if np.isfinite(start_source_row)
            else None,
            "lane_cross_TimeMS": cross_time_ms,
            "lane_cross_source_row": cross_source_row,
            "lane_change_end_TimeMS": end_time_ms,
            "lane_change_end_source_row": end_source_row,
            "event_duration_sec": (
                (end_time_ms - start_time_ms) / 1000.0
                if end_time_ms is not None and np.isfinite(start_time_ms)
                else None
            ),
        }
    )


def _compact_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    """从完整检测记录生成一行一个 event 的精简 summary。
    """
    compact_records = []
    for record in records:
        compact_record = {
            "event_id": record.get("event_id"),
            "initial_lane_id": record.get("initiator_initial_lane_id"),
            "expected_target_lane_id": record.get("expected_target_lane_id"),
            "event_start_TimeMS": record.get("event_start_TimeMS"),
            "event_start_source_row": record.get("event_start_source_row"),
            "lane_cross_TimeMS": record.get("lane_cross_TimeMS"),
            "lane_cross_source_row": record.get("lane_cross_source_row"),
            "lane_change_end_TimeMS": record.get("lane_change_end_TimeMS"),
            "lane_change_end_source_row": record.get("lane_change_end_source_row"),
            "event_duration_sec": record.get("event_duration_sec"),
            "n_frames_input": record.get("n_frames_input"),
            "n_frames_trimmed": record.get("n_frames_trimmed"),
            "initiator_v_y_at_end": record.get("initiator_v_y_at_end"),
            "initiator_a_y_at_end": record.get("initiator_a_y_at_end"),
            "end_detection_status": record.get("end_detection_status"),
            "end_method": record.get("end_method"),
            "warnings": record.get("warnings"),
        }
        compact_records.append(compact_record)
    return pd.DataFrame(compact_records, columns=COMPACT_SUMMARY_COLUMNS)


def _output_filename(input_path: Path) -> str:
    """返回单个 preliminary CSV 对应的规范 trimmed 文件名。"""
    stem = input_path.stem
    return f"{stem}_trimmed.csv" if not stem.endswith("_trimmed") else input_path.name


def process_one_dlc_window(
    file_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    """检测单个 DLC 终点、保存 trimmed CSV，并返回其 summary 记录。
    """
    input_path = Path(file_path)
    original = pd.read_csv(input_path)
    if original.empty:
        raise ValueError("无法裁剪空的 preliminary interaction window。")
    metadata = _metadata_from_frame(original, input_path)
    warnings: list[str] = []
    record: dict[str, Any] = {
        "event_id": metadata["event_id"],
        "condition": metadata["condition"],
        "scenario_id": metadata["scenario_id"],
        "input_path": str(input_path),
        "output_path": str(output_path),
        "n_frames_input": len(original),
        "duration_input_sec": _duration_sec(original),
        "stable_duration_sec": float(config["end_detection"].get("stable_duration_sec", 0.5)),
        "target_center_tolerance_m": float(config["end_detection"].get("target_center_tolerance_m", 0.3)),
        "a_y_threshold_mps2": float(config["end_detection"].get("lateral_acc_threshold_mps2", 0.10)),
        "a_y_sensitivity_threshold_mps2": float(config["end_detection"].get("sensitivity_lateral_acc_threshold_mps2", 0.05)),
        "v_y_threshold_mps": float(config["end_detection"].get("lateral_velocity_threshold_mps", 0.10)),
        "target_center_y_m": 0.0,
        "target_center_method": "ego_initial_lane_center",
        "end_method": "lane_id_kinematic",
    }
    t_cross: float | None = None
    t_end: float | None = None
    t_end_sensitivity: float | None = None
    status = "fallback"
    coordinate_frame: pd.DataFrame | None = None

    try:
        coordinate_frame, coordinate_metrics = build_temporary_dlc_coordinates_for_trimming(
            original, metadata, config
        )
        record.update(coordinate_metrics)
        initial_window = float(config["end_detection"].get("initial_lane_mode_window_sec", 0.5))
        lane_column = f"{metadata['initiator_prefix']}.LaneID{metadata['scenario_id']}"
        initiator_initial_lane = _mode_in_initial_window(coordinate_frame, lane_column, initial_window)
        ego_initial_lane = _mode_in_initial_window(coordinate_frame, EGO_LANE_COLUMN, initial_window)
        detected_target_lane = _mode_in_final_window(coordinate_frame, lane_column, initial_window)
        record.update(
            {
                "ego_initial_lane_id": ego_initial_lane,
                "initiator_initial_lane_id": initiator_initial_lane,
                "detected_target_lane_id": detected_target_lane,
            }
        )
        if initiator_initial_lane is None or ego_initial_lane is None:
            raise ValueError("缺少 Ego 或 initiator 的初始 LaneID。")
        expected_target_lane = infer_expected_target_lane(
            initiator_initial_lane, metadata["scenario_id"]
        )
        record["expected_target_lane_id"] = expected_target_lane
        lane_direction_status = (
            "pass"
            if detected_target_lane is not None and np.isclose(detected_target_lane, expected_target_lane)
            else "warning"
        )
        record["lane_direction_check_status"] = lane_direction_status
        if lane_direction_status != "pass":
            warnings.append("detected_target_lane_differs_from_expected")

        # 目标车道为 Ego 初始车道中心（y=0）。使用投影坐标中 initiator 的
        # 实际初始位置，而非强加理想 3.5 m 偏移，避免模拟器/地图微小偏差
        # 使 y-based boundary 在首帧之前被误判跨越。
        initial_lane_y = _numeric(
            coordinate_frame.loc[
                coordinate_frame["time_sec"].le(initial_window), "initiator_y_main_smooth"
            ]
        ).median()
        if not np.isfinite(initial_lane_y):
            raise ValueError("缺少 initiator 初始投影横向位置。")
        lane_boundary_y = float(initial_lane_y / 2.0)
        t_cross_lane_id = detect_t_cross_by_lane_id(
            coordinate_frame, initiator_initial_lane, lane_column
        )
        t_cross_y = detect_t_cross_by_y_main(coordinate_frame, lane_boundary_y)
        t_cross, crossing_warnings = choose_crossing_anchor(
            t_cross_lane_id, t_cross_y, config
        )
        warnings.extend(crossing_warnings)
        record.update(
            {
                "t_cross_lane_id_sec": t_cross_lane_id,
                "t_cross_y_sec": t_cross_y,
                "t_cross_sec": t_cross,
                "initiator_initial_y_main_m": float(initial_lane_y),
                "lane_boundary_y_m": lane_boundary_y,
            }
        )
        # 固定投影坐标中的目标车道中心会随道路几何产生漂移，故空间候选仅用于
        # 诊断；正式结束点由期望 LaneID 与横向运动稳定共同决定。
        t_end_spatial_030 = detect_stable_completion_after_crossing(
            coordinate_frame, t_cross, config
        )
        spatial_050_config = {
            **config,
            "end_detection": {
                **config["end_detection"],
                "target_center_tolerance_m": 0.50,
            },
        }
        t_end_spatial_050 = detect_stable_completion_after_crossing(
            coordinate_frame, t_cross, spatial_050_config
        )
        t_end = detect_lane_kinematic_completion(
            coordinate_frame,
            t_cross_lane_id,
            expected_target_lane,
            lane_column,
            config,
        )
        sensitivity_threshold = float(
            config["end_detection"].get("sensitivity_lateral_acc_threshold_mps2", 0.05)
        )
        t_end_sensitivity = detect_lane_kinematic_completion(
            coordinate_frame,
            t_cross_lane_id,
            expected_target_lane,
            lane_column,
            config,
            sensitivity_threshold,
        )
        record["t_stable_start_sec"] = t_end
        record["t_end_sec"] = t_end
        record["t_end_sensitivity_005_sec"] = t_end_sensitivity
        record["t_end_sensitivity_difference_sec"] = (
            t_end_sensitivity - t_end
            if t_end is not None and t_end_sensitivity is not None
            else None
        )
        record["t_end_spatial_030_sec"] = t_end_spatial_030
        record["t_end_spatial_050_sec"] = t_end_spatial_050
        record["t_end_spatial_030_difference_sec"] = (
            t_end_spatial_030 - t_end
            if t_end is not None and t_end_spatial_030 is not None
            else None
        )
        record["t_end_spatial_050_difference_sec"] = (
            t_end_spatial_050 - t_end
            if t_end is not None and t_end_spatial_050 is not None
            else None
        )
        if t_end is None:
            warnings.append("missing_lane_kinematic_completion_using_original_window")
            record["end_method"] = "keep_original_window"
        else:
            status = "warning" if warnings else "success"
    except (KeyError, ValueError) as error:
        warnings.append(f"detection_error:{type(error).__name__}:{error}")

    # 先从原始 preliminary window 选出正式保留帧，再附加检测过程已经计算的
    # 局部坐标与运动学特征；不在输出阶段重新投影或计算导数。
    trimmed = (
        original.loc[coordinate_frame["time_sec"].le(t_end)].copy()
        if t_end is not None and coordinate_frame is not None
        else original.copy()
    )
    trimmed = _add_trimmed_coordinate_columns(trimmed, coordinate_frame)
    write_csv(trimmed, output_path)
    record["n_frames_trimmed"] = len(trimmed)
    record["duration_trimmed_sec"] = _duration_sec(trimmed)
    record["n_frames_trimmed_sensitivity_005"] = (
        len(trim_window_to_end(coordinate_frame, t_end_sensitivity, config))
        if t_end_sensitivity is not None and coordinate_frame is not None
        else None
    )
    record["initiator_y_at_end"] = (
        float(coordinate_frame.loc[coordinate_frame["time_sec"].le(t_end), "initiator_y_main"].iloc[-1])
        if t_end is not None and coordinate_frame is not None
        else None
    )
    record["initiator_a_y_at_end"] = (
        float(coordinate_frame.loc[coordinate_frame["time_sec"].le(t_end), "initiator_a_y_main"].iloc[-1])
        if t_end is not None and coordinate_frame is not None
        else None
    )
    record["initiator_v_y_at_end"] = (
        float(coordinate_frame.loc[coordinate_frame["time_sec"].le(t_end), "initiator_v_y_main"].iloc[-1])
        if t_end is not None and coordinate_frame is not None
        else None
    )
    if t_end is None:
        record["end_method"] = "keep_original_window"
    _add_event_timing_to_record(record, original, coordinate_frame, t_cross, t_end)
    record["end_detection_status"] = status
    record["warnings"] = _join_notes(warnings)
    if diagnostics_path is not None and coordinate_frame is not None:
        create_diagnostic_plot(
            coordinate_frame,
            record,
            diagnostics_path,
            _diagnostic_axis_limits(config, metadata["scenario_id"]),
        )
    return record


def create_diagnostic_plot(
    coordinate_frame: pd.DataFrame,
    record: dict[str, Any],
    output_path: str | Path,
    axis_limits: dict[str, tuple[float, float]],
    *,
    fixed_lane_geometry: bool = False,
    lane_width_m: float = 3.5,
) -> None:
    """绘制固定坐标尺度的单个 event 诊断图。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = coordinate_frame["time_sec"]
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    scenario_id = int(record["scenario_id"])
    lane_boundary_y_m = float(record.get("lane_boundary_y_m", np.nan))
    if fixed_lane_geometry:
        target_center_display, initial_center_display = fixed_lane_geometry_display_reference(
            scenario_id, lane_width_m
        )
        # 原始 y_main 以 Ego 初始车道中心为零。仅平移到固定道路参照，不缩放
        # P08 等实际初始间距偏离 3.5 m 的 event，避免将偏差隐藏在图像中。
        y_display = np.asarray(coordinate_frame["initiator_y_main"], dtype=float) + target_center_display
        y_smooth_display = (
            np.asarray(coordinate_frame["initiator_y_main_smooth"], dtype=float)
            + target_center_display
        )
        y_label = "relative to fixed lane boundary [m]"
    else:
        target_center_display = float(record.get("target_center_y_m", 0.0)) - lane_boundary_y_m
        initial_center_display = None
        y_display = _diagnostic_y_display(
            coordinate_frame["initiator_y_main"], lane_boundary_y_m
        )
        y_smooth_display = _diagnostic_y_display(
            coordinate_frame["initiator_y_main_smooth"], lane_boundary_y_m
        )
        y_label = "relative to lane boundary [m]"
    axes[0].plot(time, y_display, alpha=0.45, label="y_main")
    axes[0].plot(time, y_smooth_display, label="y_main smoothed")
    axes[0].axhline(0.0, color="black", linestyle="--", label="lane boundary")
    axes[0].axhline(
        target_center_display,
        color="tab:green",
        linestyle="-.",
        label="target center",
    )
    axes[0].axhline(
        target_center_display,
        color="grey",
        linewidth=0.8,
        label="Ego initial center",
    )
    if initial_center_display is not None:
        axes[0].axhline(
            initial_center_display,
            color="tab:blue",
            linestyle=":",
            linewidth=0.9,
            label="initial lane center reference",
        )
    axes[0].set_ylim(axis_limits["y_relative_lane_boundary_m"])
    _mark_clipped_values(
        axes[0],
        np.concatenate([y_display, y_smooth_display]),
        axis_limits["y_relative_lane_boundary_m"],
    )
    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="best")
    axes[1].plot(time, coordinate_frame["initiator_v_y_main"], label="v_y_main")
    axes[1].axhline(
        float(record["v_y_threshold_mps"]),
        color="tab:purple",
        linestyle="--",
        label=f"{float(record['v_y_threshold_mps']):.2f} threshold",
    )
    axes[1].axhline(-float(record["v_y_threshold_mps"]), color="tab:purple", linestyle="--")
    axes[1].set_ylim(axis_limits["v_y_mps"])
    _mark_clipped_values(
        axes[1], coordinate_frame["initiator_v_y_main"].to_numpy(float), axis_limits["v_y_mps"]
    )
    axes[1].set_ylabel("lateral velocity [m/s]")
    axes[1].legend(loc="best")
    axes[2].plot(time, coordinate_frame["initiator_a_y_main"], label="a_y_main")
    axes[2].axhline(float(record["a_y_threshold_mps2"]), color="tab:red", linestyle="--", label="0.10 threshold")
    axes[2].axhline(-float(record["a_y_threshold_mps2"]), color="tab:red", linestyle="--")
    axes[2].axhline(float(record["a_y_sensitivity_threshold_mps2"]), color="tab:orange", linestyle=":", label="0.05 sensitivity")
    axes[2].axhline(-float(record["a_y_sensitivity_threshold_mps2"]), color="tab:orange", linestyle=":")
    axes[2].set_ylim(axis_limits["a_y_mps2"])
    _mark_clipped_values(
        axes[2], coordinate_frame["initiator_a_y_main"].to_numpy(float), axis_limits["a_y_mps2"]
    )
    axes[2].set_ylabel("lateral acceleration [m/s²]")
    axes[2].legend(loc="best")
    lane_column = f"{coordinate_frame['initiator_prefix'].iloc[0]}.LaneID{int(coordinate_frame['scenario_id'].iloc[0])}"
    if lane_column in coordinate_frame:
        axes[3].step(time, coordinate_frame[lane_column], where="post", label="initiator LaneID")
        axes[3].legend(loc="best")
    axes[3].set_ylim(axis_limits["lane_id"])
    axes[3].set_ylabel("LaneID")
    axes[3].set_xlabel("event-local time [s]")
    for axis in axes:
        for key, color, label in (
            ("t_cross_sec", "tab:purple", "t_cross"),
            ("t_end_sec", "tab:green", "t_end LaneID + kinematic"),
            ("t_end_sensitivity_005_sec", "tab:orange", "t_end 0.05 sensitivity"),
            ("t_end_spatial_030_sec", "tab:gray", "spatial end 0.30"),
            ("t_end_spatial_050_sec", "tab:brown", "spatial end 0.50"),
        ):
            value = record.get(key)
            if value is not None and np.isfinite(value):
                axis.axvline(float(value), color=color, linestyle="--", label=label if axis is axes[0] else None)
    figure.suptitle(
        f"{record['event_id']} | path-projection difference: "
        f"{record.get('ego_path_projection_difference_m', np.nan):.3f} m"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _create_unique_pilot_output_root(
    test_root: Path,
    stage_name: str,
    subject_id: str,
) -> Path:
    """创建不覆盖既有结果的 pilot 运行目录并返回其路径。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = test_root / stage_name
    candidate = base / f"{timestamp}_{subject_id.upper()}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{timestamp}_{subject_id.upper()}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _sha256_file(path: Path) -> str:
    """计算文件 SHA-256，用于重绘产物与输入 summary 的追溯校验。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regenerate_fixed_lane_diagnostics(
    run_directory: str | Path,
    config: dict[str, Any],
    *,
    resume: bool = False,
) -> dict[str, str | int]:
    """为既有 DLC 全量 test 重绘固定车道中心线诊断图。
    """
    run_root = Path(run_directory).resolve()
    summary_path = run_root / "dlc_end_detection_test_summary.csv"
    output_directory = run_root / "diagnostics_updated"
    if not summary_path.is_file():
        raise FileNotFoundError(f"未找到 DLC test summary：{summary_path}")
    metadata_path = output_directory / "run_metadata.json"
    if output_directory.exists() and not resume:
        raise FileExistsError(f"固定车道诊断图目录已存在，拒绝覆盖：{output_directory}")
    if output_directory.exists() and resume and metadata_path.exists():
        raise FileExistsError(f"固定车道诊断图目录已完成，拒绝覆盖：{output_directory}")

    summary = pd.read_csv(summary_path)
    required_columns = {"event_id", "input_path", "scenario_id"}
    missing_columns = sorted(required_columns.difference(summary.columns))
    if missing_columns:
        raise ValueError(f"DLC test summary 缺少字段：{', '.join(missing_columns)}")
    input_paths = [Path(value) for value in summary["input_path"]]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"重绘缺少 preliminary CSV：{missing_inputs[0]}")

    output_directory.mkdir(parents=True, exist_ok=resume)
    lane_width_m = float(config.get("lane", {}).get("lane_width_m", 3.5))
    for _, row in summary.iterrows():
        input_path = Path(row["input_path"])
        output_path = output_directory / f"{row['event_id']}_interaction_end_detection.png"
        if output_path.is_file():
            continue
        original = pd.read_csv(input_path)
        metadata = _metadata_from_frame(original, input_path)
        coordinate_frame, _ = build_temporary_dlc_coordinates_for_trimming(
            original, metadata, config
        )
        record = row.to_dict()
        create_diagnostic_plot(
            coordinate_frame,
            record,
            output_path,
            _diagnostic_axis_limits(config, int(record["scenario_id"])),
            fixed_lane_geometry=True,
            lane_width_m=lane_width_m,
        )

    # 外接盘会生成 ``._*.png`` AppleDouble 资源叉；它们不是实际诊断图，不能
    # 计入批处理完成数量。
    n_png_generated = sum(
        1
        for path in output_directory.glob("*_interaction_end_detection.png")
        if not path.name.startswith("._")
    )
    if n_png_generated != len(summary):
        raise RuntimeError(
            f"固定车道诊断图数量不完整：{n_png_generated}/{len(summary)}"
        )
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_run_directory": str(run_root),
                "source_summary_path": str(summary_path),
                "source_summary_sha256": _sha256_file(summary_path),
                "display_mode": "fixed_lane_geometry_translation_only",
                "lane_width_m": lane_width_m,
                "n_input_events": int(len(summary)),
                "n_png_generated": int(n_png_generated),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_directory": str(output_directory),
        "metadata_path": str(metadata_path),
        "n_png_generated": int(n_png_generated),
    }


def _create_unique_formal_diagnostics_root(diagnostics_root: Path) -> Path:
    """创建不覆盖既有图像的正式全量诊断目录并返回其路径。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = diagnostics_root / f"{timestamp}_all"
    suffix = 1
    while candidate.exists():
        candidate = diagnostics_root / f"{timestamp}_all_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_dlc_window_trimming(
    config: dict[str, Any],
    subject_id: str | None = None,
    diagnostics: bool = False,
    run_metadata: dict[str, Any] | None = None,
    test_all: bool = False,
) -> dict[str, Any]:
    """批量处理 DLC windows，并写出正式或 pilot 的 end-detection summary。

    ``subject_id`` 非空或 ``test_all=True`` 时视为 test：所有产物写入唯一的
    ``outputs/test`` 运行目录，绝不写入正式 interim 路径。``subject_id`` 为
    ``None`` 且 ``test_all=False`` 时写入配置指定的正式输出路径。正式运行启用
    ``diagnostics`` 时，会在配置的图像根目录下创建唯一时间戳目录。返回处理
    计数、summary 路径及产物目录。
    """
    input_config = config["input"]
    output_config = config["output"]
    input_root = Path(input_config["root"])
    formal_output_root = Path(output_config["root"])
    files = find_dlc_input_files(
        input_root, input_config["conditions"], input_config["scenario_ids"]
    )
    metadata: dict[str, Any] | None = None
    metadata_path: Path | None = None
    diagnostics_directory: Path | None = None
    if subject_id is not None or test_all:
        test_label = subject_id.upper() if subject_id is not None else "ALL"
        if subject_id is not None:
            prefix = f"{test_label}_"
            files = [path for path in files if path.name.upper().startswith(prefix)]
        output_root = _create_unique_pilot_output_root(
            Path(output_config["test_root"]), formal_output_root.name, test_label
        )
        summary_path = output_root / "dlc_end_detection_test_summary.csv"
        compact_summary_path = output_root / "dlc_end_detection_test_compact_summary.csv"
        trimmed_root = output_root / "trimmed_windows"
        diagnostic_root = output_root / "diagnostics"
        metadata = {
            "mode": "test_all" if test_all else "pilot",
            "subject_id": None if test_all else test_label,
            "run_directory": str(output_root),
            "diagnostics": diagnostics,
            **(run_metadata or {}),
        }
        metadata_path = output_root / "run_metadata.json"
    else:
        output_root = formal_output_root
        summary_path = output_root / output_config["summary_file"]
        compact_summary_path = output_root / "dlc_end_detection_compact_summary.csv"
        trimmed_root = output_root
        diagnostic_root = Path(output_config["diagnostics_root"])
        if diagnostics:
            diagnostics_directory = _create_unique_formal_diagnostics_root(diagnostic_root)
            diagnostic_root = diagnostics_directory
            metadata = {
                "mode": "formal_all",
                "diagnostics": True,
                "diagnostics_directory": str(diagnostics_directory),
                **(run_metadata or {}),
            }
            metadata_path = diagnostics_directory / "run_metadata.json"
    records: list[dict[str, Any]] = []
    for file_path in files:
        relative_parent = file_path.parent.relative_to(input_root)
        output_path = trimmed_root / relative_parent / _output_filename(file_path)
        diagnostic_path = None
        if diagnostics:
            diagnostic_path = diagnostic_root / f"{file_path.stem}_end_detection.png"
        try:
            records.append(process_one_dlc_window(file_path, output_path, config, diagnostic_path))
        except Exception as error:  # 保持批处理继续执行，并记录不可恢复错误。
            records.append(
                {
                    "event_id": file_path.stem,
                    "input_path": str(file_path),
                    "output_path": str(output_path),
                    "end_detection_status": "failed",
                    "warnings": f"batch_error:{type(error).__name__}:{error}",
                }
            )
    write_csv(pd.DataFrame(records), summary_path)
    write_csv(_compact_summary(records), compact_summary_path)
    successful_statuses = {"success"}
    report = {
        "n_input_files": len(files),
        "n_trimmed_files": sum(record.get("end_detection_status") != "failed" for record in records),
        "n_success": sum(record.get("end_detection_status") in successful_statuses for record in records),
        "n_warning": sum(record.get("end_detection_status") == "warning" for record in records),
        "n_fallback": sum(record.get("end_detection_status") == "fallback" for record in records),
        "n_failed": sum(record.get("end_detection_status") == "failed" for record in records),
        "summary_path": str(summary_path),
        "compact_summary_path": str(compact_summary_path),
        "run_directory": str(output_root) if subject_id is not None or test_all else None,
        "diagnostics_directory": str(diagnostics_directory) if diagnostics_directory else None,
    }
    if metadata is not None and metadata_path is not None:
        metadata.update(report)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return report
