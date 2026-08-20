"""Persistent prediction statistics.

Kept in a single JSON file so the counters survive a restart of the app. Writes
go to a temporary file first and are then moved into place, so an interrupted
write can never leave a half-written file behind and lose the history.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone

from src import config

_LOCK = threading.Lock()
_HISTORY_LIMIT = 400  # keeps the file small; the totals below are never trimmed

_EMPTY: dict = {"version": 1, "total": 0, "first_seen": None, "models": {}, "history": []}


def _read() -> dict:
    path = config.STATS_PATH
    if not path.exists():
        return json.loads(json.dumps(_EMPTY))
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return json.loads(json.dumps(_EMPTY))

    for key, default in _EMPTY.items():
        stored.setdefault(key, default)
    return stored


def _write(payload: dict) -> None:
    path = config.STATS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    )
    try:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def load() -> dict:
    with _LOCK:
        return _read()


def record(model_key: str, model_name: str, family: str, prediction: float) -> None:
    """Add one prediction to the running totals."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _LOCK:
        payload = _read()
        payload["total"] = int(payload["total"]) + 1
        payload["first_seen"] = payload["first_seen"] or now

        entry = payload["models"].setdefault(
            model_key,
            {
                "name": model_name,
                "family": family,
                "runs": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
                "last_used": None,
                "last_value": None,
            },
        )
        entry["name"] = model_name
        entry["family"] = family
        entry["runs"] = int(entry["runs"]) + 1
        entry["sum"] = float(entry["sum"]) + prediction
        entry["min"] = prediction if entry["min"] is None else min(float(entry["min"]), prediction)
        entry["max"] = prediction if entry["max"] is None else max(float(entry["max"]), prediction)
        entry["last_used"] = now
        entry["last_value"] = round(prediction, 2)

        payload["history"].append(
            {"at": now, "model": model_key, "value": round(prediction, 2)}
        )
        payload["history"] = payload["history"][-_HISTORY_LIMIT:]

        _write(payload)


def reset() -> None:
    with _LOCK:
        _write(json.loads(json.dumps(_EMPTY)))


def per_model(payload: dict) -> list[dict]:
    """One row per model, ready for a table or a bar chart."""
    rows = []
    for key, entry in payload["models"].items():
        runs = int(entry["runs"]) or 1
        rows.append(
            {
                "key": key,
                "Model": entry["name"],
                "Family": config.FAMILIES.get(entry["family"], {}).get(
                    "label", entry["family"]
                ),
                "Runs": int(entry["runs"]),
                "Mean MPa": round(float(entry["sum"]) / runs, 2),
                "Min MPa": round(float(entry["min"]), 2) if entry["min"] is not None else None,
                "Max MPa": round(float(entry["max"]), 2) if entry["max"] is not None else None,
                "Last used": (entry["last_used"] or "").replace("T", " ")[:16],
            }
        )
    return sorted(rows, key=lambda row: row["Runs"], reverse=True)


def overall(payload: dict) -> dict:
    """Totals across every model."""
    runs = sum(int(e["runs"]) for e in payload["models"].values())
    total_sum = sum(float(e["sum"]) for e in payload["models"].values())
    values = [entry["value"] for entry in payload["history"]]
    return {
        "total": int(payload["total"]),
        "models_used": len(payload["models"]),
        "mean": round(total_sum / runs, 2) if runs else float("nan"),
        "last_value": values[-1] if values else None,
        "since": (payload["first_seen"] or "").replace("T", " ")[:16],
    }
