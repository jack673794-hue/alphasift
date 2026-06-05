# -*- coding: utf-8 -*-
"""Persistence helpers for screen runs and evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from alphasift.models import EvaluationResult, Pick, PickEvaluation, ScreenResult

_RUN_METADATA_SUFFIX = ".meta"


def save_screen_result(
    result: ScreenResult,
    *,
    data_dir: Path,
    path: str | Path | None = None,
    jsonl: bool = False,
) -> Path:
    """Persist a screen result and return the written path."""
    output_path = Path(path) if path is not None else data_dir / "runs" / f"{result.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.saved_path = str(output_path)
    if jsonl:
        output_path.write_text("\n".join(screen_result_to_jsonl(result)) + "\n", encoding="utf-8")
    else:
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _write_screen_result_metadata(result, output_path)
    return output_path


def load_screen_result(run_ref: str | Path, *, data_dir: Path) -> ScreenResult:
    """Load a saved screen result by run_id or path."""
    path = resolve_run_path(run_ref, data_dir=data_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    pick_items = data.get("picks", [])
    pick_fields = {field.name for field in fields(Pick)}
    result_fields = {field.name for field in fields(ScreenResult)}
    data["picks"] = [
        Pick(**{key: value for key, value in item.items() if key in pick_fields})
        for item in pick_items
        if isinstance(item, dict)
    ]
    filtered = {key: value for key, value in data.items() if key in result_fields}
    loaded = ScreenResult(**filtered)
    loaded.saved_path = str(path)
    return loaded


def save_evaluation_result(
    result: EvaluationResult,
    *,
    data_dir: Path,
    path: str | Path | None = None,
    jsonl: bool = False,
) -> Path:
    output_path = Path(path) if path is not None else data_dir / "evaluations" / f"{result.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.saved_path = str(output_path)
    if jsonl:
        output_path.write_text("\n".join(evaluation_result_to_jsonl(result)) + "\n", encoding="utf-8")
    else:
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return output_path


def list_saved_runs(
    *,
    data_dir: Path,
    limit: int = 20,
    strategy: str | None = None,
) -> list[dict[str, object]]:
    runs_dir = data_dir / "runs"
    if not runs_dir.is_dir():
        return []
    limit = int(limit)
    if limit <= 0:
        return []
    items = []
    for path in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        metadata = _read_run_metadata(path)
        if metadata is None:
            continue
        if strategy and metadata.get("strategy") != strategy:
            continue
        items.append(metadata)
        if len(items) >= limit:
            break
    return items


def _write_screen_result_metadata(result: ScreenResult, output_path: Path) -> None:
    metadata_path = _run_metadata_path(output_path)
    metadata = {
        "run_id": result.run_id or output_path.stem,
        "strategy": result.strategy,
        "market": result.market,
        "created_at": result.created_at,
        "picks": len(result.picks),
        "path": str(output_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_metadata_path(run_path: Path) -> Path:
    return run_path.with_suffix(run_path.suffix + _RUN_METADATA_SUFFIX)


def _read_run_metadata(path: Path) -> dict[str, object] | None:
    metadata_path = _run_metadata_path(path)
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                return _normalize_run_metadata(metadata, path)
        except Exception:
            pass
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _normalize_run_metadata(data, path)


def _normalize_run_metadata(data: dict, path: Path) -> dict[str, object]:
    picks = data.get("picks", [])
    if isinstance(picks, list):
        pick_count = len(picks)
    else:
        try:
            pick_count = int(picks)
        except (TypeError, ValueError):
            pick_count = 0
    return {
        "run_id": data.get("run_id", path.stem),
        "strategy": data.get("strategy", ""),
        "market": data.get("market", ""),
        "created_at": data.get("created_at", ""),
        "picks": pick_count,
        "path": str(path),
    }


def resolve_run_path(run_ref: str | Path, *, data_dir: Path) -> Path:
    path = Path(run_ref)
    if path.is_file():
        return path
    candidate = data_dir / "runs" / f"{run_ref}.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Saved run not found: {run_ref}")


def screen_result_to_jsonl(result: ScreenResult) -> list[str]:
    data = asdict(result)
    picks = data.pop("picks", [])
    lines = [json.dumps({"type": "run", **data}, ensure_ascii=False)]
    for pick in picks:
        lines.append(json.dumps({"type": "pick", "run_id": result.run_id, **pick}, ensure_ascii=False))
    return lines


def evaluation_result_to_jsonl(result: EvaluationResult) -> list[str]:
    data = asdict(result)
    picks = data.pop("picks", [])
    lines = [json.dumps({"type": "evaluation", **data}, ensure_ascii=False)]
    for pick in picks:
        lines.append(json.dumps({"type": "pick_evaluation", "run_id": result.run_id, **pick}, ensure_ascii=False))
    return lines


def evaluation_from_dict(data: dict) -> EvaluationResult:
    pick_fields = {field.name for field in fields(PickEvaluation)}
    result_fields = {field.name for field in fields(EvaluationResult)}
    data = dict(data)
    data["picks"] = [
        PickEvaluation(**{key: value for key, value in item.items() if key in pick_fields})
        for item in data.get("picks", [])
        if isinstance(item, dict)
    ]
    return EvaluationResult(**{key: value for key, value in data.items() if key in result_fields})
