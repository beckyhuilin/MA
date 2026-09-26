"""DLC 直线路段共享坐标构造与诊断。
"""

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


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """将指定列转换为数值，并将无效值保留为 NaN。"""
    return pd.to_numeric(frame[column], errors="coerce")


def _first_valid_position(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> int:
    """返回所有指定列均为有限值的首个行位置。"""
    valid = np.ones(len(frame), dtype=bool)
    for column in columns:
        valid &= np.isfinite(_numeric(frame, column).to_numpy(dtype=float))
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise ValueError(f"不存在所有指定列均为有限值的行：{', '.join(columns)}")
    return int(positions[0])


def estimate_main_road_direction(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """根据 Ego 前 1 秒轨迹估计前进向量 u与右侧法向量n。
    """
    required = (TIME_COLUMN, EGO_X_COLUMN, EGO_Y_COLUMN)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少坐标列：{', '.join(missing)}")

    time_sec = _numeric(frame, TIME_COLUMN).to_numpy(dtype=float) / 1000.0
    points = frame[[EGO_X_COLUMN, EGO_Y_COLUMN]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(time_sec) & np.isfinite(points).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("至少需要两个有效的 Ego Global X/Y 位置。")

    first_time = time_sec[valid][0]
    direction_window_sec = float(config.get("coordinate", {}).get("direction_window_sec", 1.0))
    window = valid & (time_sec <= first_time + direction_window_sec)
    if window.sum() < 2:
        window = valid
    window_points = points[window]
    centered = window_points - window_points.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= np.finfo(float).eps:
        raise ValueError("Ego 初始轨迹无法估计有效方向。")
    u = right_vectors[0]
    if np.dot(u, window_points[-1] - window_points[0]) < 0:
        u = -u
    u = u / np.linalg.norm(u)
    n = np.array([u[1], -u[0]])
    return u, n


def compute_dlc_origin(
    frame: pd.DataFrame,
    u: np.ndarray,
    n: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """返回 Ego 初始车道中心的 Global X/Y 原点，单位为米。
    """
    del u, config  # 原点只由右侧法向量决定。
    required = (EGO_X_COLUMN, EGO_Y_COLUMN, EGO_LATERAL_DISTANCE_COLUMN)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少原点构造列：{', '.join(missing)}")
    position = _first_valid_position(frame, required)
    ego_point = frame.loc[frame.index[position], [EGO_X_COLUMN, EGO_Y_COLUMN]].astype(float)
    lateral_distance = float(_numeric(frame, EGO_LATERAL_DISTANCE_COLUMN).iloc[position])
    return ego_point.to_numpy() - lateral_distance * n


def project_global_to_main_road(
    x_global: pd.Series | np.ndarray,
    y_global: pd.Series | np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将 Global X/Y 投影到共享主路 ``x_main/y_main`` 坐标，单位为米。"""
    points = np.column_stack(
        [pd.to_numeric(x_global, errors="coerce"), pd.to_numeric(y_global, errors="coerce")]
    ).astype(float)
    offsets = points - origin
    return offsets @ u, offsets @ n


def smooth_lateral_position(y_main: pd.Series | np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """仅为终点检测轻度平滑横向位置，不改变原始坐标。"""
    values = pd.to_numeric(pd.Series(y_main), errors="coerce").astype(float)
    smoothing = config.get("smoothing", {})
    if not smoothing.get("enabled", True):
        return values.to_numpy()
    filled = values.interpolate(limit_direction="both")
    if filled.isna().any():
        return values.to_numpy()
    method = smoothing.get("method", "savgol")
    if method == "rolling_mean":
        window = int(smoothing.get("window_length_frames", 5))
        return filled.rolling(window, center=True, min_periods=1).mean().to_numpy()
    if method != "savgol":
        raise ValueError(f"不支持的平滑方法：{method!r}")

    requested_window = int(smoothing.get("window_length_frames", 11))
    polyorder = int(smoothing.get("polyorder", 2))
    window = min(requested_window, len(filled) if len(filled) % 2 else len(filled) - 1)
    if window <= polyorder or window < 3:
        return filled.to_numpy()
    return savgol_filter(filled.to_numpy(), window_length=window, polyorder=polyorder)


def compute_lateral_kinematics(
    y_main: pd.Series | np.ndarray,
    time_sec: pd.Series | np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回平滑横向位置、横向速度与横向加速度，均使用 SI 单位。
    """
    del config
    y_smooth = np.asarray(y_main, dtype=float)
    time = np.asarray(time_sec, dtype=float)
    if len(y_smooth) != len(time):
        raise ValueError("y_main 与 time_sec 的长度必须一致。")
    if len(time) < 3:
        return y_smooth, np.full(len(time), np.nan), np.full(len(time), np.nan)
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("计算导数时 TimeMS 必须为有限值且严格递增。")
    velocity = np.gradient(y_smooth, time)
    acceleration = np.gradient(velocity, time)
    return y_smooth, velocity, acceleration


def _initiator_columns(metadata: dict[str, Any]) -> tuple[str, str]:
    """返回单个 DLC event 中动态解析的 initiator Global X/Y 列名。"""
    prefix = str(metadata["initiator_prefix"])
    return f"{prefix}.X", f"{prefix}.Y"


def _orient_right_normal_from_lane_ids(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    n: np.ndarray,
) -> np.ndarray:
    """校准 ``n``，使 DLC LaneID 增大方向对应正（右侧）y。
    """
    if "scenario_id" not in metadata:
        return n
    initiator_x, initiator_y = _initiator_columns(metadata)
    lane_column = f"{metadata['initiator_prefix']}.LaneID{metadata['scenario_id']}"
    required = (EGO_X_COLUMN, EGO_Y_COLUMN, initiator_x, initiator_y, EGO_LANE_COLUMN, lane_column)
    if any(column not in frame for column in required):
        return n
    position = _first_valid_position(frame, required)
    ego = frame.loc[frame.index[position], [EGO_X_COLUMN, EGO_Y_COLUMN]].astype(float).to_numpy()
    initiator = frame.loc[frame.index[position], [initiator_x, initiator_y]].astype(float).to_numpy()
    lane_difference = float(_numeric(frame, lane_column).iloc[position] - _numeric(frame, EGO_LANE_COLUMN).iloc[position])
    lateral_difference = float(np.dot(initiator - ego, n))
    return -n if lane_difference * lateral_difference < 0 else n


def build_temporary_dlc_coordinates_for_trimming(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """添加临时 DLC 坐标，并返回不阻断流程的验证指标。
    """
    initiator_x_column, initiator_y_column = _initiator_columns(metadata)
    required = (TIME_COLUMN, EGO_X_COLUMN, EGO_Y_COLUMN, initiator_x_column, initiator_y_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少临时坐标构造列：{', '.join(missing)}")

    u, n = estimate_main_road_direction(frame, config)
    n = _orient_right_normal_from_lane_ids(frame, metadata, n)
    origin = compute_dlc_origin(frame, u, n, config)
    result = frame.copy()
    result["time_sec"] = (_numeric(result, TIME_COLUMN) - _numeric(result, TIME_COLUMN).iloc[0]) / 1000.0
    result["ego_x_main"], result["ego_y_main"] = project_global_to_main_road(
        result[EGO_X_COLUMN], result[EGO_Y_COLUMN], origin, u, n
    )
    result["initiator_x_main"], result["initiator_y_main"] = project_global_to_main_road(
        result[initiator_x_column], result[initiator_y_column], origin, u, n
    )
    result["initiator_y_main_smooth"] = smooth_lateral_position(
        result["initiator_y_main"], config
    )
    _, velocity, acceleration = compute_lateral_kinematics(
        result["initiator_y_main_smooth"], result["time_sec"], config
    )
    result["initiator_v_y_main"] = velocity
    result["initiator_a_y_main"] = acceleration
    return result, coordinate_validation_metrics(result, config)


def coordinate_validation_metrics(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float | None]:
    """计算直线路段坐标检查指标，供 summary 与 pilot 图使用。"""
    lane_width = float(config.get("lane", {}).get("lane_width_m", 3.5))
    ego_xy = frame[[EGO_X_COLUMN, EGO_Y_COLUMN]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid_pairs = np.isfinite(ego_xy).all(axis=1)
    path_length = float(np.linalg.norm(np.diff(ego_xy[valid_pairs], axis=0), axis=1).sum()) if valid_pairs.sum() > 1 else None
    x_main = pd.to_numeric(frame["ego_x_main"], errors="coerce").dropna()
    projection_displacement = float(x_main.iloc[-1] - x_main.iloc[0]) if len(x_main) > 1 else None

    lateral_rmse: float | None = None
    lateral_max_error: float | None = None
    if EGO_LANE_COLUMN in frame and EGO_LATERAL_DISTANCE_COLUMN in frame:
        lane = _numeric(frame, EGO_LANE_COLUMN)
        lateral_distance = _numeric(frame, EGO_LATERAL_DISTANCE_COLUMN)
        initial_lane_values = lane.dropna()
        if not initial_lane_values.empty:
            initial_lane = float(initial_lane_values.iloc[0])
            expected = (lane - initial_lane) * lane_width + lateral_distance
            error = pd.to_numeric(frame["ego_y_main"], errors="coerce") - expected
            error = error.dropna()
            if not error.empty:
                lateral_rmse = float(np.sqrt(np.mean(np.square(error))))
                lateral_max_error = float(error.abs().max())

    return {
        "ego_path_length_m": path_length,
        "ego_projection_displacement_m": projection_displacement,
        "ego_path_projection_difference_m": (
            path_length - projection_displacement
            if path_length is not None and projection_displacement is not None
            else None
        ),
        "ego_lateral_validation_rmse_m": lateral_rmse,
        "ego_lateral_validation_max_abs_error_m": lateral_max_error,
    }
