#!/usr/bin/env python3
"""运行 DLC interaction-end detection：支持 pilot 或确认后的全量批处理。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hmm_interaction.window_trimming_dlc import run_dlc_window_trimming  # noqa: E402


def _resolve_project_path(value: str) -> str:
    """将 YAML 路径解析为项目根目录相对路径，保证 CLI 可复现运行。"""
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _load_yaml(path: Path) -> dict:
    """读取一个 YAML 配置文件并返回字典。"""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> int:
    """解析 pilot/全量批处理参数并运行 DLC 裁剪。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/window_trimming_dlc.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--subject", default="P01", help="Pilot subject ID（默认：P01）。")
    mode.add_argument("--all", action="store_true", help="处理全部 DLC events。")
    parser.add_argument(
        "--confirm-pilot",
        action="store_true",
        help="在人工确认 pilot 后，使用 --all 时必须提供此参数。",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="与 --all 一同使用，将全量测试产物写入 outputs/test 而非正式路径。",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="为每个处理的 event 生成一张 end-detection 诊断图。",
    )
    args = parser.parse_args()
    if args.all and not (args.confirm_pilot or args.test):
        parser.error("人工确认 pilot 后，--all 必须同时提供 --confirm-pilot。")
    if args.confirm_pilot and not args.all:
        parser.error("--confirm-pilot 仅能与 --all 一同使用。")
    if args.test and not args.all:
        parser.error("--test 仅能与 --all 一同使用。")
    if args.test and args.confirm_pilot:
        parser.error("--test 与 --confirm-pilot 不能同时使用。")

    config = _load_yaml(Path(_resolve_project_path(args.config)))
    config["input"]["root"] = _resolve_project_path(config["input"]["root"])
    config["output"]["root"] = _resolve_project_path(config["output"]["root"])
    config["output"]["diagnostics_root"] = _resolve_project_path(
        config["output"]["diagnostics_root"]
    )
    config["output"]["test_root"] = _resolve_project_path(config["output"]["test_root"])
    report = run_dlc_window_trimming(
        config,
        None if args.all else args.subject.upper(),
        args.diagnostics,
        {
            "command": " ".join(sys.argv),
            "config_path": _resolve_project_path(args.config),
        },
        args.test,
    )
    print(
        "Processed {n_input_files} DLC windows: {n_success} success, {n_warning} "
        "warning, {n_fallback} fallback, {n_failed} failed. Summary: {summary_path}".format(
            **report
        )
    )
    print(f"精简 summary: {report['compact_summary_path']}")
    if report["run_directory"] is not None:
        print(f"Pilot 结果目录: {report['run_directory']}")
    if report["diagnostics_directory"] is not None:
        print(f"正式诊断图目录: {report['diagnostics_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
