"""MLC Scenario 1 的 Map1 中心线读取与横向几何重构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolylineProjection:
    """记录点投影至中心线后的几何结果，距离单位为 m。"""

    point: np.ndarray
    tangent: np.ndarray
    signed_right_offset_m: float


# 将指定列转换为数值，无效值保留为 NaN。
def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


# 读取 Map1 的五条中心线，并将地图坐标转换为车辆 Global 坐标。
def load_map1_centerlines(path: str | Path) -> dict[str, np.ndarray]:
    raw = pd.read_csv(path, sep=";", decimal=",", skiprows=[0])
    if raw.shape[1] < 10:
        raise ValueError("Map1 中心线文件必须包含五条 Spur 的 X/Y 列。")
    centerlines: dict[str, np.ndarray] = {}
    for spur, column_index in zip(("spur1", "spur2", "spur3", "spur4", "spur5"), range(0, 10, 2)):
        points = (
            raw.iloc[:, [column_index, column_index + 1]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy(float, copy=True)
        )
        if len(points) < 2:
            raise ValueError(f"Map1 {spur} 不包含至少两个有效坐标点。")
        # Map1 的 Y 轴与车辆 Global Y 方向相反：x 保持不变，y 取负。
        points[:, 1] *= -1.0
        centerlines[spur] = points
    return centerlines


# 将 Scenario 1 initiator 的确定性 raw LaneID 映射为物理组和 Map1 参考 Spur。
def map_lane_id_to_reference_spur(lane_id: float | int | None, config: dict[str, Any]) -> tuple[str, str | None, str]:
    if lane_id is None or not np.isfinite(float(lane_id)):
        return "unmapped", None, "unmapped"
    value = int(round(float(lane_id)))
    mapping = config["map"]["lane_mapping"]
    groups = (
        ("lane5", "spur1_lane5", "spur1"),
        ("lane6", "spur2_lane6", "spur2"),
        ("lane7", "spur3_lane7", "spur3"),
        ("ramp", "spur5_merging", "spur5"),
    )
    for group, key, spur in groups:
        if value in {int(item) for item in mapping[key]}:
            return group, spur, "stable_reference"
    if value in {int(item) for item in mapping["transition"]}:
        return "transition", None, "transition"
    return "unmapped", None, "unmapped"


# 将一个 Global 坐标点投影到 polyline 最近线段，并返回相对车辆前进方向右侧为正的横向偏移。
def project_point_to_spur(point: np.ndarray, centerline: np.ndarray, forward_hint: np.ndarray) -> PolylineProjection:
    starts = centerline[:-1]
    vectors = centerline[1:] - starts
    squared = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared > np.finfo(float).eps
    if not valid.any():
        raise ValueError("中心线不存在有效线段。")
    starts, vectors, squared = starts[valid], vectors[valid], squared[valid]
    relative = point - starts
    fraction = np.clip(np.einsum("ij,ij->i", relative, vectors) / squared, 0.0, 1.0)
    projected = starts + fraction[:, None] * vectors
    index = int(np.argmin(np.linalg.norm(projected - point, axis=1)))
    tangent = vectors[index] / np.linalg.norm(vectors[index])
    # 统一中心线切向量与 Ego 前进方向，确保右法向量的符号一致。
    if np.dot(tangent, forward_hint) < 0:
        tangent = -tangent
    # Global Y 与标准数学坐标方向相反；在该左手平面中物理右侧为 [-t_y, t_x]。
    right_normal = np.array([-tangent[1], tangent[0]])
    offset = float(np.dot(point - projected[index], right_normal))
    return PolylineProjection(projected[index], tangent, offset)


# 为完整 preliminary window 追加 Map1 reference、所有诊断偏移和正式重构横向偏移字段。
def reconstruct_initiator_lateral_offset(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    centerlines: dict[str, np.ndarray],
    forward_hint: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    prefix = str(metadata["initiator_prefix"])
    x_column, y_column, lane_column = f"{prefix}.X", f"{prefix}.Y", f"{prefix}.LaneID1"
    missing = [column for column in (x_column, y_column, lane_column) if column not in frame]
    if missing:
        raise ValueError(f"缺少 Scenario 1 Map 重构列：{', '.join(missing)}")
    result = frame.copy()
    lane = numeric(result, lane_column).to_numpy(float)
    points = result[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    offset_columns = {spur: np.full(len(result), np.nan) for spur in ("spur1", "spur2", "spur3", "spur5")}
    groups: list[str] = []
    references: list[str | None] = []
    statuses: list[str] = []
    reconstructed = np.full(len(result), np.nan)
    for index, point in enumerate(points):
        group, reference, status = map_lane_id_to_reference_spur(lane[index], config)
        groups.append(group)
        references.append(reference)
        statuses.append(status)
        if not np.isfinite(point).all():
            continue
        for spur in offset_columns:
            offset_columns[spur][index] = project_point_to_spur(point, centerlines[spur], forward_hint).signed_right_offset_m
        if reference is not None:
            reconstructed[index] = offset_columns[reference][index]
    result["initiator_raw_lane_id"] = lane
    result["initiator_physical_lane_group"] = groups
    result["initiator_reference_spur"] = references
    result["reconstruction_status"] = statuses
    for spur, values in offset_columns.items():
        result[f"initiator_offset_to_{spur}"] = values
    result["initiator_lateral_offset_lane_center_reconstructed"] = reconstructed
    return result
