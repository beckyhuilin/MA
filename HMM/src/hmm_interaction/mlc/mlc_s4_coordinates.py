from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


TIME_COLUMN = "TimeMS"
EGO_X_COLUMN = "Ego.Global.X"
EGO_Y_COLUMN = "Ego.Global.Y"
EGO_LATERAL_DISTANCE_COLUMN = "Ego.LateralDistance"
EGO_LANE_COLUMN = "Ego.LaneIdx (Lane ID of Ego)"


# 将指定列转换为数值，无效值保留为 NaN。
def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


# 返回指定列均有效的第一帧位置。
def first_valid_position(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    valid = np.ones(len(frame), dtype=bool)
    for column in columns:
        if column not in frame:
            raise ValueError(f"缺少坐标构造列：{column}")
        valid &= np.isfinite(numeric(frame, column).to_numpy(float))
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise ValueError(f"不存在所有指定列均为有限值的行：{', '.join(columns)}")
    return int(positions[0])


# 使用 Ego 前 1 秒 Global X/Y 轨迹估计主路前进 u 与候选右法向量 n。
def estimate_main_road_direction(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    required = (TIME_COLUMN, EGO_X_COLUMN, EGO_Y_COLUMN)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少坐标列：{', '.join(missing)}")
    time = numeric(frame, TIME_COLUMN).to_numpy(float) / 1000.0
    points = frame[[EGO_X_COLUMN, EGO_Y_COLUMN]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(time) & np.isfinite(points).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("至少需要两个有效的 Ego Global X/Y 位置。")
    first_time = time[valid][0]
    duration = float(config.get("coordinate", {}).get("direction_window_sec", 1.0))
    window = valid & (time <= first_time + duration)
    if window.sum() < 2:
        window = valid
    window_points = points[window]
    centered = window_points - window_points.mean(axis=0)
    _, values, vectors = np.linalg.svd(centered, full_matrices=False)
    if values[0] <= np.finfo(float).eps:
        raise ValueError("Ego 初始轨迹无法估计有效方向。")
    forward = vectors[0]
    if np.dot(forward, window_points[-1] - window_points[0]) < 0:
        forward = -forward
    forward = forward / np.linalg.norm(forward)
    return forward, np.array([forward[1], -forward[0]])


# 按起始标准 LaneID 校正法向量，使车道编号增大方向为 shared y 的正方向。
def orient_right_normal_from_lane_ids(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    normal: np.ndarray,
) -> np.ndarray:
    prefix = str(metadata["initiator_prefix"])
    lane_column = f"{prefix}.LaneID4"
    required = (EGO_X_COLUMN, EGO_Y_COLUMN, f"{prefix}.X", f"{prefix}.Y", EGO_LANE_COLUMN, lane_column)
    if any(column not in frame for column in required):
        return normal
    position = first_valid_position(frame, required)
    ego_lane = float(numeric(frame, EGO_LANE_COLUMN).iloc[position])
    initiator_lane = float(numeric(frame, lane_column).iloc[position])
    if ego_lane not in {5.0, 6.0, 7.0} or initiator_lane not in {5.0, 6.0, 7.0}:
        return normal
    ego = frame.loc[frame.index[position], [EGO_X_COLUMN, EGO_Y_COLUMN]].astype(float).to_numpy()
    initiator = frame.loc[frame.index[position], [f"{prefix}.X", f"{prefix}.Y"]].astype(float).to_numpy()
    lateral_difference = float(np.dot(initiator - ego, normal))
    return -normal if (initiator_lane - ego_lane) * lateral_difference < 0 else normal


# 用 Ego 首个有效位置与 LateralDistance 回推其初始车道中心点作为共享坐标原点。
def compute_ego_initial_lane_origin(frame: pd.DataFrame, normal: np.ndarray) -> np.ndarray:
    required = (EGO_X_COLUMN, EGO_Y_COLUMN, EGO_LATERAL_DISTANCE_COLUMN)
    position = first_valid_position(frame, required)
    ego_point = frame.loc[frame.index[position], [EGO_X_COLUMN, EGO_Y_COLUMN]].astype(float).to_numpy()
    lateral_distance = float(numeric(frame, EGO_LATERAL_DISTANCE_COLUMN).iloc[position])
    return ego_point - lateral_distance * normal


# 将 Global X/Y 投影为以 Ego 初始车道中心为原点的 shared x_main/y_main。
def project_global_to_main_road(
    x_global: pd.Series,
    y_global: pd.Series,
    origin: np.ndarray,
    forward: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([pd.to_numeric(x_global, errors="coerce"), pd.to_numeric(y_global, errors="coerce")]).astype(float)
    offsets = points - origin
    return offsets @ forward, offsets @ normal


# 对横向位置实施与 DLC 一致的 Savgol 平滑，仅供运动学与终点检测使用。
def smooth_lateral_position(y_main: pd.Series, config: dict[str, Any]) -> np.ndarray:
    values = pd.to_numeric(y_main, errors="coerce").astype(float)
    settings = config.get("smoothing", {})
    if not settings.get("enabled", True):
        return values.to_numpy()
    filled = values.interpolate(limit_direction="both")
    if filled.isna().any():
        return values.to_numpy()
    if settings.get("method", "savgol") != "savgol":
        raise ValueError("MLC Scenario 4 仅支持 savgol 横向平滑。")
    requested = int(settings.get("window_length_frames", 11))
    polyorder = int(settings.get("polyorder", 2))
    window = min(requested, len(filled) if len(filled) % 2 else len(filled) - 1)
    if window <= polyorder or window < 3:
        return filled.to_numpy()
    return savgol_filter(filled.to_numpy(), window_length=window, polyorder=polyorder)


# 使用实际 TimeMS 对平滑横向位置求一、二阶导数。
def compute_lateral_kinematics(
    y_smooth: np.ndarray,
    time_sec: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_sec, dtype=float)
    values = np.asarray(y_smooth, dtype=float)
    if len(time) != len(values):
        raise ValueError("y_main 与 time_sec 的长度必须一致。")
    if len(time) < 3:
        return np.full(len(time), np.nan), np.full(len(time), np.nan)
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("计算导数时 TimeMS 必须为有限值且严格递增。")
    velocity = np.gradient(values, time)
    return velocity, np.gradient(velocity, time)


# 构造 Scenario 4 检测与写入所需的 shared-coordinate 和 initiator 横向运动学字段。
def build_mlc_s4_coordinates(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    prefix = str(metadata["initiator_prefix"])
    initiator_columns = (f"{prefix}.X", f"{prefix}.Y")
    initiator_lane_column = f"{prefix}.LaneID4"
    required = (
        TIME_COLUMN,
        EGO_X_COLUMN,
        EGO_Y_COLUMN,
        EGO_LATERAL_DISTANCE_COLUMN,
        EGO_LANE_COLUMN,
        *initiator_columns,
        initiator_lane_column,
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少 MLC 临时坐标构造列：{', '.join(missing)}")
    time_ms = numeric(frame, TIME_COLUMN).to_numpy(float)
    if not np.isfinite(time_ms).all() or np.any(np.diff(time_ms) <= 0):
        raise ValueError("MLC Scenario 4 的 TimeMS 必须为有限值且严格递增。")
    forward, normal = estimate_main_road_direction(frame, config)
    normal = orient_right_normal_from_lane_ids(frame, metadata, normal)
    origin = compute_ego_initial_lane_origin(frame, normal)
    result = frame.copy()
    result["time_sec"] = (numeric(result, TIME_COLUMN) - numeric(result, TIME_COLUMN).iloc[0]) / 1000.0
    result["ego_x_main"], result["ego_y_main"] = project_global_to_main_road(
        result[EGO_X_COLUMN], result[EGO_Y_COLUMN], origin, forward, normal
    )
    result["initiator_x_main"], result["initiator_y_main"] = project_global_to_main_road(
        result[initiator_columns[0]], result[initiator_columns[1]], origin, forward, normal
    )
    result["initiator_y_main_smooth"] = smooth_lateral_position(result["initiator_y_main"], config)
    result["initiator_v_y_main"], result["initiator_a_y_main"] = compute_lateral_kinematics(
        result["initiator_y_main_smooth"].to_numpy(float), result["time_sec"]
    )
    # 原始 LaneID 单独标准化列名；三位数值保留为拓扑诊断，不参与物理车道映射。
    result["initiator_raw_lane_id"] = numeric(result, initiator_lane_column)
    return result, normal
