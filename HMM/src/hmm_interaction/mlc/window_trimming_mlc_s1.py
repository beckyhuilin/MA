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
from map1_centerline_reconstruction import (
    load_map1_centerlines,
    numeric,
    reconstruct_initiator_lateral_offset,
)
from mlc_s1_coordinates import (
    add_map_reference_shared_coordinate,
    build_mlc_s1_coordinates,
    estimate_main_road_direction,
)


MLC_S1_RECONSTRUCTION_COLUMNS = (
    "initiator_raw_lane_id",
    "initiator_physical_lane_group",
    "initiator_reference_spur",
    "reconstruction_status",
    "initiator_offset_to_spur1",
    "initiator_offset_to_spur2",
    "initiator_offset_to_spur3",
    "initiator_offset_to_spur5",
    "initiator_lateral_offset_lane_center_reconstructed",
)

MLC_S1_DERIVED_COLUMNS = (
    "time_sec",
    "ego_x_main",
    "ego_y_main",
    "initiator_x_main",
    "initiator_y_main",
    "initiator_y_main_smooth",
    "initiator_v_y_main",
    "initiator_a_y_main",
    "initiator_raw_lane_id",
    "initiator_reference_center_y_main",
    "initiator_y_main_map_residual_m",
)

MLC_S1_SUMMARY_COLUMNS = (
    "event_id",
    "subject_id",
    "condition",
    "n_frames_preliminary",
    "n_frames_reconstructed",
    "n_frames_retrimmed",
    "t_start_candidate_sec",
    "t_start_sec",
    "start_method",
    "stable_duration_sec",
    "t_target_lane_entry_sec",
    "t_warning_revised_lane161_sec",
    "t_end_sec",
    "t_end_after_start_sec",
    "t_end_sensitivity_005_sec",
    "t_end_spatial_020_sec",
    "t_end_spatial_030_sec",
    "t_end_spatial_050_sec",
    "end_method",
    "event_start_TimeMS",
    "event_start_source_row",
    "target_lane_entry_TimeMS",
    "target_lane_entry_source_row",
    "lane_change_end_TimeMS",
    "lane_change_end_source_row",
    "raw_lane_id_at_start",
    "raw_lane_id_at_end",
    "initiator_y_main_at_end",
    "initiator_v_y_main_at_end",
    "initiator_a_y_main_at_end",
    "initiator_offset_to_spur3_at_end",
    "abs_initiator_offset_to_spur3_at_end",
    "shared_coordinate_mae_m",
    "shared_coordinate_rmse_m",
    "shared_coordinate_max_abs_error_m",
    "shared_coordinate_correlation",
    "n_map_validation_frames",
    "reconstruction_status",
    "status",
    "warnings",
    "input_path",
    "reconstructed_output_path",
    "retrimmed_output_path",
)


# 返回由 condition/MLC/1 目录组成的正常 Scenario 1 preliminary 文件。
def find_mlc_s1_input_files(input_root: str | Path, conditions: Iterable[str]) -> list[Path]:
    root = Path(input_root)
    files: list[Path] = []
    for condition in conditions:
        files.extend(
            path
            for path in (root / condition / "MLC" / "1").glob("*_interaction.csv")
            if not path.name.startswith("._")
        )
    return sorted(files)


# 从 preliminary CSV 读取 Scenario 1 必需 metadata，并验证 event 类型。
def metadata_from_frame(frame: pd.DataFrame, file_path: Path) -> dict[str, Any]:
    required = ("event_id", "condition", "scenario_id", "initiator_prefix")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"MLC Scenario 1 缺少 event metadata 列：{', '.join(missing)}")
    first = frame.iloc[0]
    scenario_id = pd.to_numeric(pd.Series([first["scenario_id"]]), errors="coerce").iloc[0]
    if not np.isfinite(scenario_id) or int(scenario_id) != 1:
        raise ValueError(f"MLC trimming 仅支持 scenario_id 1，当前为 {first['scenario_id']}。")
    return {
        "event_id": str(first["event_id"]),
        "condition": str(first["condition"]),
        "scenario_id": 1,
        "initiator_prefix": str(first["initiator_prefix"]),
        "input_path": str(file_path),
    }


# 在布尔条件中返回首个实际时间长度不少于 duration 的连续区间起点。
def first_continuous_window_start(time_sec: np.ndarray, condition: np.ndarray, duration_sec: float) -> float | None:
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


# 返回 raw LaneID 131 的首帧，仅作为道路几何诊断候选。
def detect_s1_start_candidate(frame: pd.DataFrame, metadata: dict[str, Any], config: dict[str, Any]) -> int | None:
    lane_column = f"{metadata['initiator_prefix']}.LaneID1"
    if lane_column not in frame:
        raise ValueError(f"缺少 Scenario 1 initiator LaneID 列：{lane_column}")
    lane = numeric(frame, lane_column).to_numpy(float)
    target = float(config["start_detection"]["lane_id"])
    positions = np.flatnonzero(np.isfinite(lane) & np.isclose(lane, target))
    return int(positions[0]) if len(positions) else None


# 在 t_start 后寻找 initiator 首次进入定义的 raw Lane 7 segment 的时刻。
def detect_s1_target_lane_entry(frame: pd.DataFrame, config: dict[str, Any]) -> float | None:
    lane = numeric(frame, "initiator_raw_lane_id").to_numpy(float)
    time = numeric(frame, "time_sec").to_numpy(float)
    target_ids = np.asarray(config["end_detection"]["target_lane_raw_ids"], dtype=float)
    eligible = np.isfinite(lane) & np.isfinite(time) & np.isin(np.rint(lane), target_ids)
    positions = np.flatnonzero(eligible)
    return float(time[positions[0]]) if len(positions) else None


# 在已进入 target Lane 7 后寻找横向速度、加速度均连续稳定的最早窗口。
def detect_s1_stable_completion(
    frame: pd.DataFrame,
    target_entry_sec: float | None,
    config: dict[str, Any],
    acceleration_threshold: float | None = None,
) -> float | None:
    if target_entry_sec is None:
        return None
    settings = config["end_detection"]
    acceleration_limit = float(settings["lateral_acc_threshold_mps2"] if acceleration_threshold is None else acceleration_threshold)
    time = numeric(frame, "time_sec").to_numpy(float)
    velocity = numeric(frame, "initiator_v_y_main").to_numpy(float)
    acceleration = numeric(frame, "initiator_a_y_main").to_numpy(float)
    stable = (
        (time >= target_entry_sec)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & (np.abs(velocity) < float(settings["lateral_velocity_threshold_mps"]))
        & (np.abs(acceleration) < acceleration_limit)
    )
    return first_continuous_window_start(time, stable, float(settings["stable_duration_sec"]))


# 在首次 LaneID 161 后寻找 warning 专用的连续低速、低加速度窗口。
def detect_s1_warning_revised_completion(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[float | None, float | None]:
    settings = config["warning_revised"]
    time = numeric(frame, "time_sec").to_numpy(float)
    lane = numeric(frame, "initiator_raw_lane_id").to_numpy(float)
    positions = np.flatnonzero(np.isfinite(time) & np.isfinite(lane) & np.isclose(lane, float(settings["start_lane_id"])))
    if len(positions) == 0:
        return None, None
    search_start = float(time[positions[0]])
    velocity = numeric(frame, "initiator_v_y_main").to_numpy(float)
    acceleration = numeric(frame, "initiator_a_y_main").to_numpy(float)
    stable = (
        (time >= search_start)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & (np.abs(velocity) < float(settings["lateral_velocity_threshold_mps"]))
        & (np.abs(acceleration) < float(settings["lateral_acc_threshold_mps2"]))
    )
    return search_start, first_continuous_window_start(time, stable, float(settings["stable_duration_sec"]))


# 返回 target Lane 7 entry 后首次满足指定空间容差的诊断候选，不参与正式终点判断。
def detect_s1_spatial_candidate(frame: pd.DataFrame, target_entry_sec: float | None, tolerance_m: float) -> float | None:
    if target_entry_sec is None:
        return None
    time = numeric(frame, "time_sec")
    y_main = numeric(frame, "initiator_y_main")
    eligible = time.ge(target_entry_sec) & y_main.abs().lt(tolerance_m)
    return float(time.loc[eligible].iloc[0]) if eligible.any() else None


# 根据正式或插值时刻返回其对应的原始 TimeMS 和 source_row，终点使用不晚于目标时刻的实际帧。
def time_and_source_row_at_or_before(frame: pd.DataFrame, time_sec: float | None) -> tuple[float | None, int | None]:
    if time_sec is None:
        return None, None
    eligible = numeric(frame, "time_sec").le(time_sec)
    if not eligible.any():
        return None, None
    row = frame.loc[eligible].iloc[-1]
    return float(numeric(pd.DataFrame([row]), "TimeMS").iloc[0]), int(row["source_row"])


# 计算与 Map1 reference center shared y 的可比较残差指标，只覆盖 131/150 和 Lane 7 区间。
def shared_coordinate_validation_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float | int | None]:
    lane = numeric(frame, "initiator_raw_lane_id").to_numpy(float)
    residual = numeric(frame, "initiator_y_main_map_residual_m").to_numpy(float)
    y_main = numeric(frame, "initiator_y_main").to_numpy(float)
    reference_y = numeric(frame, "initiator_reference_center_y_main").to_numpy(float)
    valid_ids = np.asarray(
        [*config["validation"]["parallel_ramp_raw_ids"], *config["end_detection"]["target_lane_raw_ids"]], dtype=float
    )
    selected = np.isin(np.rint(lane), valid_ids) & np.isfinite(residual) & np.isfinite(y_main) & np.isfinite(reference_y)
    if not selected.any():
        return {
            "shared_coordinate_mae_m": None,
            "shared_coordinate_rmse_m": None,
            "shared_coordinate_max_abs_error_m": None,
            "shared_coordinate_correlation": None,
            "n_map_validation_frames": 0,
        }
    values = residual[selected]
    compared_y = y_main[selected]
    compared_reference = reference_y[selected]
    # 任一序列没有变化时相关系数没有定义，保留为空而不是触发数值告警。
    correlation = (
        np.corrcoef(compared_y, compared_reference)[0, 1]
        if len(values) > 1 and np.std(compared_y) > np.finfo(float).eps and np.std(compared_reference) > np.finfo(float).eps
        else np.nan
    )
    return {
        "shared_coordinate_mae_m": float(np.mean(np.abs(values))),
        "shared_coordinate_rmse_m": float(np.sqrt(np.mean(values**2))),
        "shared_coordinate_max_abs_error_m": float(np.max(np.abs(values))),
        "shared_coordinate_correlation": float(correlation) if np.isfinite(correlation) else None,
        "n_map_validation_frames": int(len(values)),
    }


# 从 start 至 end 写出正式 retrimmed window；终点不含任何未经确认的后续帧。
def trim_s1_interaction_window(frame: pd.DataFrame, t_end_after_start_sec: float) -> pd.DataFrame:
    return frame.loc[numeric(frame, "time_sec").le(t_end_after_start_sec)].copy()


# 绘制图片
def create_mlc_s1_diagnostic_plot(frame: pd.DataFrame, record: dict[str, Any], output_path: str | Path, config: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = numeric(frame, "time_sec").to_numpy(float)
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    axes[0].plot(time, numeric(frame, "initiator_y_main"), color="tab:blue", alpha=0.45, label="y_main")
    axes[0].plot(time, numeric(frame, "initiator_y_main_smooth"), color="tab:orange", label="y_main smoothed")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1, label="Ego initial lane center (origin)")
    configured_limit = max(abs(value) for value in config["diagnostics"]["axis_limits"]["y_main_m"])
    observed_limit = np.nanmax(np.abs(numeric(frame, "initiator_y_main_smooth").to_numpy(float)))
    y_limit = max(configured_limit, observed_limit * 1.05) if np.isfinite(observed_limit) else configured_limit
    axes[0].set_ylim(-y_limit, y_limit)
    axes[0].set_ylabel("shared lateral position [m]")
    axes[0].legend(loc="best")
    axes[1].plot(time, numeric(frame, "initiator_v_y_main"), color="tab:blue", label="v_y_main")
    threshold_v = float(config["end_detection"]["lateral_velocity_threshold_mps"])
    axes[1].axhline(threshold_v, color="tab:purple", linestyle="--")
    axes[1].axhline(-threshold_v, color="tab:purple", linestyle="--", label=f"{threshold_v:.2f} velocity threshold")
    if record["status"] == "warning_revised":
        warning_threshold_v = float(config["warning_revised"]["lateral_velocity_threshold_mps"])
        axes[1].axhline(warning_threshold_v, color="gray", linestyle="-.")
        axes[1].axhline(-warning_threshold_v, color="gray", linestyle="-.", label=f"{warning_threshold_v:.2f} warning threshold")
    axes[1].set_ylim(config["diagnostics"]["axis_limits"]["v_y_mps"])
    axes[1].set_ylabel("lateral velocity [m/s]")
    axes[1].legend(loc="best")
    axes[2].plot(time, numeric(frame, "initiator_a_y_main"), color="tab:blue", label="a_y_main")
    threshold_a = float(config["end_detection"]["lateral_acc_threshold_mps2"])
    axes[2].axhline(threshold_a, color="tab:red", linestyle="--")
    axes[2].axhline(-threshold_a, color="tab:red", linestyle="--", label=f"{threshold_a:.2f} primary threshold")
    if record["status"] == "fallback":
        fallback_threshold_a = float(config["fallback"]["lateral_acc_threshold_mps2"])
        axes[2].axhline(fallback_threshold_a, color="gray", linestyle="-.")
        axes[2].axhline(-fallback_threshold_a, color="gray", linestyle="-.", label=f"{fallback_threshold_a:.2f} fallback threshold")
    if record["status"] == "warning_revised":
        warning_threshold_a = float(config["warning_revised"]["lateral_acc_threshold_mps2"])
        axes[2].axhline(warning_threshold_a, color="gray", linestyle="-.")
        axes[2].axhline(-warning_threshold_a, color="gray", linestyle="-.", label=f"{warning_threshold_a:.2f} warning threshold")
    axes[2].axhline(float(config["end_detection"]["sensitivity_lateral_acc_threshold_mps2"]), color="tab:orange", linestyle=":")
    axes[2].axhline(-float(config["end_detection"]["sensitivity_lateral_acc_threshold_mps2"]), color="tab:orange", linestyle=":", label=f"{float(config['end_detection']['sensitivity_lateral_acc_threshold_mps2']):.2f} sensitivity")
    axes[2].set_ylim(config["diagnostics"]["axis_limits"]["a_y_mps2"])
    axes[2].set_ylabel("lateral acceleration [m/s²]")
    axes[2].legend(loc="best")
    axes[3].plot(time, numeric(frame, "initiator_raw_lane_id"), color="tab:blue", label="raw LaneID")
    axes[3].set_ylabel("raw LaneID")
    axes[3].set_xlabel("event-local time from interaction start [s]")
    axes[3].legend(loc="best")
    target_entry = record.get("t_target_lane_entry_after_start_sec")
    end_time = record.get("t_end_after_start_sec")
    for axis in axes:
        axis.axvline(0.0, color="tab:blue", linestyle="--", label="interaction t_start" if axis is axes[0] else None)
        if target_entry is not None and np.isfinite(target_entry):
            axis.axvline(target_entry, color="tab:green", linestyle="--", label="target Lane 7 entry" if axis is axes[0] else None)
        if end_time is not None and np.isfinite(end_time):
            axis.axvline(end_time, color="tab:orange", linestyle="--", linewidth=2, label="final t_end" if axis is axes[0] else None)
    axes[0].legend(loc="best")
    title_end = "not detected" if end_time is None or not np.isfinite(end_time) else f"{end_time:.3f} s"
    fig.suptitle(f"{record['event_id']} | {record['status']} | {record['end_method']} | final t_end={title_end}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 从既有 Stage A 文件读取原始信号执行 Stage B；Map1 字段只用于验证。
def evaluate_mlc_s1_window(
    file_path: str | Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    path = Path(file_path)
    reconstructed_path = Path(config["output"]["reconstruct_lateral_root"]) / f"{path.stem}_reconstructed.csv"
    reconstructed = pd.read_csv(reconstructed_path)
    metadata = metadata_from_frame(reconstructed, reconstructed_path)
    centerlines = load_map1_centerlines(config["map"]["centerline_file"])
    start_position = detect_s1_start_candidate(reconstructed, metadata, config)
    record: dict[str, Any] = {
        "event_id": metadata["event_id"],
        "subject_id": str(reconstructed["subject_id"].iloc[0]),
        "condition": metadata["condition"],
        "n_frames_preliminary": int(len(reconstructed)),
        "n_frames_reconstructed": int(len(reconstructed)),
        "n_frames_retrimmed": None,
        "t_start_candidate_sec": float((numeric(reconstructed, "TimeMS").iloc[start_position] - numeric(reconstructed, "TimeMS").iloc[0]) / 1000.0) if start_position is not None else None,
        "t_start_sec": 0.0,
        "start_method": "predefined_interaction_window_start",
        "stable_duration_sec": float(config["end_detection"]["stable_duration_sec"]),
        "status": "warning",
        "end_method": "not_detected",
        "input_path": str(reconstructed_path),
        "reconstructed_output_path": str(reconstructed_path),
        "reconstruction_status": "success",
        "warnings": "",
    }
    coordinate_frame, coordinate = build_mlc_s1_coordinates(reconstructed, metadata, config)
    coordinate_frame = add_map_reference_shared_coordinate(coordinate_frame, metadata, centerlines, coordinate)
    record["event_start_TimeMS"], record["event_start_source_row"] = time_and_source_row_at_or_before(coordinate_frame, 0.0)
    record["raw_lane_id_at_start"] = float(coordinate_frame["initiator_raw_lane_id"].iloc[0])
    target_entry = detect_s1_target_lane_entry(coordinate_frame, config)
    t_primary = detect_s1_stable_completion(coordinate_frame, target_entry, config)
    t_fallback = (
        detect_s1_stable_completion(
            coordinate_frame,
            target_entry,
            config,
            float(config["fallback"]["lateral_acc_threshold_mps2"]),
        )
        if target_entry is not None and t_primary is None
        else None
    )
    warning_search_start, t_warning_revised = (
        detect_s1_warning_revised_completion(coordinate_frame, config)
        if target_entry is not None and t_primary is None and t_fallback is None
        else (None, None)
    )
    t_end = t_primary if t_primary is not None else t_fallback if t_fallback is not None else t_warning_revised
    t_sensitivity = detect_s1_stable_completion(
        coordinate_frame, target_entry, config, float(config["end_detection"]["sensitivity_lateral_acc_threshold_mps2"])
    )
    record.update(shared_coordinate_validation_metrics(coordinate_frame, config))
    if record["n_map_validation_frames"] == 0:
        record["warnings"] = "map_validation_no_comparable_frames"
    record["t_target_lane_entry_after_start_sec"] = target_entry
    record["t_target_lane_entry_sec"] = target_entry
    record["t_warning_revised_lane161_sec"] = warning_search_start
    record["t_end_after_start_sec"] = t_end
    record["t_end_sec"] = t_end
    record["t_end_sensitivity_005_sec"] = t_sensitivity
    for tolerance in config["spatial_diagnostics"]["target_center_tolerances_m"]:
        key = int(round(float(tolerance) * 100))
        candidate = detect_s1_spatial_candidate(coordinate_frame, target_entry, float(tolerance))
        record[f"t_end_spatial_{key:03d}_sec"] = candidate
    if target_entry is None:
        record["warnings"] = ";".join(filter(None, [record["warnings"], "missing_target_lane7_entry"]))
        return reconstructed, coordinate_frame, None, record
    record["target_lane_entry_TimeMS"], record["target_lane_entry_source_row"] = time_and_source_row_at_or_before(coordinate_frame, target_entry)
    if t_primary is None:
        record["warnings"] = ";".join(filter(None, [record["warnings"], "missing_strict_stable_target_lane_window"]))
    if t_primary is None and t_fallback is None:
        record["warnings"] = ";".join(filter(None, [record["warnings"], "missing_relaxed_stable_target_lane_window"]))
    if t_end is None:
        record["warnings"] = ";".join(filter(None, [record["warnings"], "missing_warning_revised_stable_window"]))
        return reconstructed, coordinate_frame, None, record
    trimmed = trim_s1_interaction_window(coordinate_frame, t_end)
    end_row = trimmed.iloc[-1]
    record.update(
        {
            "n_frames_retrimmed": int(len(trimmed)),
            "end_method": "stable_target_lane" if t_primary is not None else "relaxed_acceleration_stable_window" if t_fallback is not None else "post_lane161_relaxed_kinematics",
            "status": "success" if t_primary is not None else "fallback" if t_fallback is not None else "warning_revised",
            "lane_change_end_TimeMS": float(end_row["TimeMS"]),
            "lane_change_end_source_row": int(end_row["source_row"]),
            "raw_lane_id_at_end": float(end_row["initiator_raw_lane_id"]),
            "initiator_y_main_at_end": float(end_row["initiator_y_main"]),
            "initiator_v_y_main_at_end": float(end_row["initiator_v_y_main"]),
            "initiator_a_y_main_at_end": float(end_row["initiator_a_y_main"]),
            "initiator_offset_to_spur3_at_end": float(end_row["initiator_offset_to_spur3"]),
            "abs_initiator_offset_to_spur3_at_end": abs(float(end_row["initiator_offset_to_spur3"])),
        }
    )
    return reconstructed, coordinate_frame, trimmed, record


# 仅执行 Stage A：保留完整窗口并计算 initiator 相对当前参考 Spur 中心线的横向偏移。
def evaluate_mlc_s1_stage_a(file_path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(file_path)
    original = pd.read_csv(path)
    metadata = metadata_from_frame(original, path)
    forward_full = estimate_main_road_direction(original, config)
    centerlines = load_map1_centerlines(config["map"]["centerline_file"])
    reconstructed = reconstruct_initiator_lateral_offset(original, metadata, centerlines, forward_full, config)
    return reconstructed, {
        "event_id": metadata["event_id"],
        "condition": metadata["condition"],
        "n_frames_preliminary": int(len(original)),
        "n_frames_reconstructed": int(len(reconstructed)),
        "n_frames_retrimmed": None,
        "reconstruction_status": "success",
        "status": "success",
        "end_method": "not_applicable_stage_a_only",
        "warnings": "",
        "input_path": str(path),
    }


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


# 将无法读取或构造的 event 记录为 failed，且不写出伪造的输出文件。
def failed_record(file_path: Path, error: Exception) -> dict[str, Any]:
    return {
        "event_id": file_path.stem.removesuffix("_interaction"),
        "status": "failed",
        "reconstruction_status": "failed",
        "end_method": "not_written_invalid_input",
        "warnings": str(error),
        "input_path": str(file_path),
    }


# 批量执行独立 Stage A 或读取已有 Stage A 结果执行 Stage B。
def run_mlc_s1_window_trimming(
    config: dict[str, Any],
    subject_id: str | None = None,
    diagnostics: bool = False,
    run_metadata: dict[str, Any] | None = None,
    test_mode: bool = False,
    stage_a_only: bool = False,
    event_ids: set[str] | None = None,
) -> dict[str, Any]:
    files = find_mlc_s1_input_files(config["input"]["root"], config["input"]["conditions"])
    if subject_id is not None:
        files = [path for path in files if path.stem.startswith(f"{subject_id}_")]
    if event_ids is not None:
        files = [path for path in files if path.stem.removesuffix("_interaction") in event_ids]
        found = {path.stem.removesuffix("_interaction") for path in files}
        if found != event_ids:
            raise FileNotFoundError(f"未找到指定的 MLC Scenario 1 event：{', '.join(sorted(event_ids - found))}")
    if not files:
        raise FileNotFoundError("未找到符合条件的 MLC Scenario 1 preliminary window。")
    output = config["output"]
    label = "warning_selected" if event_ids is not None else "all" if subject_id is None else subject_id
    run_root = create_unique_output_root(output["test_root"], label) if (test_mode or subject_id is not None) else None
    if run_root is not None:
        reconstructed_root = run_root / "reconstruct lateral"
        retrimmed_root = run_root / "retrimmed"
        summary_path = run_root / ("mlc_s1_reconstruction_test_summary.csv" if stage_a_only else "mlc_s1_retrimming_test_summary.csv")
        diagnostic_root = run_root / "diagnostics" if diagnostics and not stage_a_only else None
    else:
        reconstructed_root = Path(output["reconstruct_lateral_root"])
        retrimmed_root = Path(output["retrimmed_root"])
        summary_path = Path(output["reconstruction_summary_path"] if stage_a_only else output["summary_path"])
        diagnostic_root = create_unique_output_root(output["diagnostics_root"], "all") if diagnostics and not stage_a_only else None
    records: list[dict[str, Any]] = []
    for path in files:
        try:
            if stage_a_only:
                reconstructed, record = evaluate_mlc_s1_stage_a(path, config)
                coordinate_frame, trimmed = None, None
            else:
                reconstructed, coordinate_frame, trimmed, record = evaluate_mlc_s1_window(path, config)
            if stage_a_only:
                reconstructed_path = reconstructed_root / f"{path.stem}_reconstructed.csv"
                write_csv(reconstructed, reconstructed_path)
                record["reconstructed_output_path"] = str(reconstructed_path)
            if trimmed is not None:
                destination = retrimmed_root if run_root is not None else retrimmed_root / record["status"]
                retrimmed_path = destination / f"{path.stem}_retrimmed.csv"
                write_csv(trimmed, retrimmed_path)
                record["retrimmed_output_path"] = str(retrimmed_path)
            if diagnostic_root is not None and coordinate_frame is not None:
                # 诊断始终画到 t_start 后窗口末尾，便于核对 t_end 后是否仍存在横向动作。
                create_mlc_s1_diagnostic_plot(
                    coordinate_frame,
                    record,
                    diagnostic_root / record["status"] / f"{path.stem}_retrimming.png",
                    config,
                )
            records.append(record)
        except (ValueError, KeyError, OSError) as error:
            records.append(failed_record(path, error))
    full_summary = pd.DataFrame(records)
    summary = full_summary.reindex(columns=MLC_S1_SUMMARY_COLUMNS)
    write_csv(summary, summary_path)
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_input_files": len(files),
        "summary_path": str(summary_path),
        "diagnostics": diagnostics,
        "stage_a_only": stage_a_only,
        "event_ids": sorted(event_ids) if event_ids is not None else None,
        **(run_metadata or {}),
    }
    if run_root is not None:
        metadata["run_directory"] = str(run_root)
        (run_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    elif diagnostic_root is not None:
        metadata["diagnostics_directory"] = str(diagnostic_root)
        (diagnostic_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    statuses = full_summary.get("status", pd.Series(dtype=object))
    return {
        "n_input_files": len(files),
        "n_success": int(statuses.eq("success").sum()),
        "n_fallback": int(statuses.eq("fallback").sum()),
        "n_warning_revised": int(statuses.eq("warning_revised").sum()),
        "n_warning": int(statuses.eq("warning").sum()),
        "n_failed": int(statuses.eq("failed").sum()),
        "summary_path": str(summary_path),
        "run_directory": str(run_root) if run_root is not None else None,
        "diagnostics_directory": str(diagnostic_root) if diagnostic_root is not None else None,
    }
