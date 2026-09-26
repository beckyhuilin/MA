#!/usr/bin/env python3
"""运行无地图中心线的 MLC Scenario 4 interaction-end detection。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLC_SOURCE_ROOT = PROJECT_ROOT / "src" / "hmm_interaction" / "mlc"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(MLC_SOURCE_ROOT))

from window_trimming_mlc_s4 import run_mlc_s4_window_trimming  # noqa: E402


# 将 YAML 中的相对路径解析为项目根目录下的绝对路径。
def resolve_project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


# 读取 MLC Scenario 4 裁剪配置。
def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# 解析运行模式，并将 pilot、test 与正式全量安全地路由至不同输出目录。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/window_trimming_mlc_s4.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--subject", default="P01", help="Pilot subject ID（默认：P01）。")
    mode.add_argument("--all", action="store_true", help="处理全部 MLC Scenario 4 events。")
    parser.add_argument("--confirm-pilot", action="store_true", help="正式全量运行必须提供。")
    parser.add_argument("--test", action="store_true", help="全量测试写入 outputs/test，不改正式数据。")
    parser.add_argument("--diagnostics", action="store_true", help="生成 end-detection 诊断图。")
    args = parser.parse_args()
    if args.all and not (args.confirm_pilot or args.test):
        parser.error("--all 必须同时提供 --test 或 --confirm-pilot。")
    if args.confirm_pilot and not args.all:
        parser.error("--confirm-pilot 仅能与 --all 一同使用。")
    if args.test and not args.all:
        parser.error("--test 仅能与 --all 一同使用。")
    if args.test and args.confirm_pilot:
        parser.error("--test 与 --confirm-pilot 不能同时使用。")
    config_path = Path(resolve_project_path(args.config))
    config = load_yaml(config_path)
    config["input"]["root"] = resolve_project_path(config["input"]["root"])
    for key in ("trimmed_root", "diagnostics_root", "test_root"):
        config["output"][key] = resolve_project_path(config["output"][key])
    report = run_mlc_s4_window_trimming(
        config,
        None if args.all else args.subject.upper(),
        args.diagnostics,
        {"command": " ".join(sys.argv), "config_path": str(config_path)},
        args.test,
    )
    print(
        "Processed {n_input_files} MLC Scenario 4 windows: {n_success} success, "
        "{n_warning} warning, {n_warning_revised} warning-revised, {n_fallback} fallback, "
        "{n_failed} failed. Summary: {summary_path}".format(
            **report
        )
    )
    if report["run_directory"] is not None:
        print(f"Test 结果目录: {report['run_directory']}")
    if report["diagnostics_directory"] is not None:
        print(f"诊断图目录: {report['diagnostics_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
