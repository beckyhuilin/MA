from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from map1_centerline_reconstruction import numeric, project_point_to_spur

TIME_COLUMN = "TimeMS"
EGO_X_COLUMN = "Ego.Global.X"
EGO_Y_COLUMN = "Ego.Global.Y"
EGO_LATERAL_DISTANCE_COLUMN = "Ego.LateralDistance"


@dataclass(frozen=True)
class SharedCoordinate:
    """记录 shared-coordinate 的原点、前进轴和右侧法向量。"""

    origin: np.ndarray
    forward: np.ndarray
    right_normal: np.ndarray


# 返回所有指定列均为有限数值的首个行位置。
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


# 复制 DLC 的 Ego 前 1 秒 PCA 主方向构造，并以首尾位移确认车辆前进正向。
def estimate_main_road_direction(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
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
    duration = float(config["coordinate"]["direction_window_sec"])
    use = valid & (time <= first_time + duration)
    if use.sum() < 2:
        use = valid
    selected = points[use]
    centered = selected - selected.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= np.finfo(float).eps:
        raise ValueError("Ego 初始轨迹无法估计有效方向。")
    forward = vectors[0]
    if np.dot(forward, selected[-1] - selected[0]) < 0:
        forward = -forward
    return forward / np.linalg.norm(forward)


# 用 Ego 首个有效 Global 位置及 LateralDistance 回推其初始车道中心点。
def compute_ego_initial_lane_origin(frame: pd.DataFrame, right_normal: np.ndarray) -> np.ndarray:
    position = first_valid_position(frame, (EGO_X_COLUMN, EGO_Y_COLUMN, EGO_LATERAL_DISTANCE_COLUMN))
    ego_point = frame.loc[frame.index[position], [EGO_X_COLUMN, EGO_Y_COLUMN]].astype(float).to_numpy()
    lateral_distance = float(numeric(frame, EGO_LATERAL_DISTANCE_COLUMN).iloc[position])
    return ego_point - lateral_distance * right_normal


# 将 Global X/Y 数组投影为以 Ego 初始 lane 7 中心为原点的 shared x_main/y_main。
def project_global_to_main_road(
    x_global: pd.Series | np.ndarray,
    y_global: pd.Series | np.ndarray,
    coordinate: SharedCoordinate,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([pd.to_numeric(x_global, errors="coerce"), pd.to_numeric(y_global, errors="coerce")]).astype(float)
    offsets = points - coordinate.origin
    return offsets @ coordinate.forward, offsets @ coordinate.right_normal


# 使用 DLC 同一套 Savgol 参数平滑横向位置，仅供运动学和终点检测使用。
def smooth_lateral_position(y_main: pd.Series, config: dict[str, Any]) -> np.ndarray:
    values = pd.to_numeric(y_main, errors="coerce").astype(float)
    settings = config["smoothing"]
    if not settings.get("enabled", True):
        return values.to_numpy()
    filled = values.interpolate(limit_direction="both")
    if filled.isna().any():
        return values.to_numpy()
    requested = int(settings["window_length_frames"])
    polyorder = int(settings["polyorder"])
    window = min(requested, len(filled) if len(filled) % 2 else len(filled) - 1)
    if window <= polyorder or window < 3:
        return filled.to_numpy()
    return savgol_filter(filled.to_numpy(), window_length=window, polyorder=polyorder)


# 按实际 TimeMS 对平滑横向位置计算一阶速度和二阶加速度。
def compute_lateral_kinematics(y_smooth: np.ndarray, time_sec: pd.Series) -> tuple[np.ndarray, np.ndarray]:
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


# 从正式 t_start 开始构造 Scenario 1 共享坐标、平滑横向位置和横向运动学字段。
def build_mlc_s1_coordinates(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, SharedCoordinate]:
    prefix = str(metadata["initiator_prefix"])
    required = (TIME_COLUMN, EGO_X_COLUMN, EGO_Y_COLUMN, EGO_LATERAL_DISTANCE_COLUMN, f"{prefix}.X", f"{prefix}.Y", f"{prefix}.LaneID1")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"缺少 Scenario 1 坐标构造列：{', '.join(missing)}")
    time_ms = numeric(frame, TIME_COLUMN).to_numpy(float)
    if not np.isfinite(time_ms).all() or np.any(np.diff(time_ms) <= 0):
        raise ValueError("Scenario 1 的 TimeMS 必须为有限值且严格递增。")
    forward = estimate_main_road_direction(frame, config)
    # 保留 Scenario 1 已校准的物理右侧为正；三位数 LaneID 不用于方向翻转。
    right_normal = np.array([-forward[1], forward[0]])
    coordinate = SharedCoordinate(
        origin=compute_ego_initial_lane_origin(frame, right_normal),
        forward=forward,
        right_normal=right_normal,
    )
    result = frame.copy()
    result["time_sec"] = (numeric(result, TIME_COLUMN) - numeric(result, TIME_COLUMN).iloc[0]) / 1000.0
    result["ego_x_main"], result["ego_y_main"] = project_global_to_main_road(result[EGO_X_COLUMN], result[EGO_Y_COLUMN], coordinate)
    result["initiator_x_main"], result["initiator_y_main"] = project_global_to_main_road(result[f"{prefix}.X"], result[f"{prefix}.Y"], coordinate)
    result["initiator_y_main_smooth"] = smooth_lateral_position(result["initiator_y_main"], config)
    result["initiator_v_y_main"], result["initiator_a_y_main"] = compute_lateral_kinematics(
        result["initiator_y_main_smooth"].to_numpy(float), result["time_sec"]
    )
    result["initiator_raw_lane_id"] = numeric(result, f"{prefix}.LaneID1")
    return result, coordinate


# 将 Map1 最近中心点投影至同一 shared coordinate，形成与 initiator_y_main 可直接比较的验证残差。
def add_map_reference_shared_coordinate(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    centerlines: dict[str, np.ndarray],
    coordinate: SharedCoordinate,
) -> pd.DataFrame:
    prefix = str(metadata["initiator_prefix"])
    points = frame[[f"{prefix}.X", f"{prefix}.Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    reference_y = np.full(len(frame), np.nan)
    residual = np.full(len(frame), np.nan)
    for position, point in enumerate(points):
        reference = frame["initiator_reference_spur"].iloc[position]
        if reference not in centerlines or not np.isfinite(point).all():
            continue
        projected = project_point_to_spur(point, centerlines[reference], coordinate.forward).point
        reference_y[position] = float(np.dot(projected - coordinate.origin, coordinate.right_normal))
        residual[position] = float(frame["initiator_y_main"].iloc[position] - reference_y[position])
    result = frame.copy()
    result["initiator_reference_center_y_main"] = reference_y
    result["initiator_y_main_map_residual_m"] = residual
    return result
