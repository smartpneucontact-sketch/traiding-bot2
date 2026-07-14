"""Pre-registered forward-test protocol — loader + integrity hash.

The paper test is only credible evidence if the success/kill criteria were
committed BEFORE the first funded rebalance and provably never edited
afterwards. `protocol.json` (repo root) is the machine-readable contract;
its sha256 is recorded into the model's state at the first funded
rebalance (core/runner.py). Any later mismatch means the protocol was
modified mid-test and the run must be permanently flagged
MODIFIED_AFTER_START by the scorer (Phase A2).

Deliberately dependency-free (json + hashlib only): this module must be
importable from the live order path, the dashboard, and offline tooling
without dragging in pandas/yfinance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
PROTOCOL_JSON_PATH = _BASE_DIR / "protocol.json"
PROTOCOL_MD_PATH = _BASE_DIR / "PROTOCOL.md"

# Every criterion must carry exactly these keys (extras like
# `and_conditions` are allowed — the 6mo regime AND-condition needs one).
REQUIRED_CRITERION_FIELDS = (
    "id", "description", "metric", "op", "threshold", "checkpoint", "action",
)
ALLOWED_OPS = ("<", "<=", ">", ">=", "==", "between")
ALLOWED_CHECKPOINTS = ("continuous", "3mo", "6mo", "12mo")


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file's bytes. Chunked so it stays O(1) memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_criterion(crit: dict) -> None:
    missing = [f for f in REQUIRED_CRITERION_FIELDS if f not in crit]
    if missing:
        raise ValueError(
            f"protocol criterion {crit.get('id', '<no id>')!r} missing "
            f"required fields: {missing}"
        )
    if crit["op"] not in ALLOWED_OPS:
        raise ValueError(
            f"protocol criterion {crit['id']!r}: unknown op {crit['op']!r} "
            f"(allowed: {ALLOWED_OPS})"
        )
    if crit["checkpoint"] not in ALLOWED_CHECKPOINTS:
        raise ValueError(
            f"protocol criterion {crit['id']!r}: unknown checkpoint "
            f"{crit['checkpoint']!r} (allowed: {ALLOWED_CHECKPOINTS})"
        )
    thr = crit["threshold"]
    if crit["op"] == "between":
        if (not isinstance(thr, (list, tuple)) or len(thr) != 2
                or not all(isinstance(x, (int, float)) for x in thr)
                or thr[0] > thr[1]):
            raise ValueError(
                f"protocol criterion {crit['id']!r}: 'between' needs a "
                f"[lo, hi] numeric pair, got {thr!r}"
            )
    elif not isinstance(thr, (int, float)):
        raise ValueError(
            f"protocol criterion {crit['id']!r}: threshold must be numeric "
            f"for op {crit['op']!r}, got {thr!r}"
        )


def load_protocol(path: str | Path | None = None) -> dict:
    """Load + validate protocol.json.

    Returns the parsed dict with a computed `_sha256` key added (leading
    underscore = derived, not part of the committed file) so callers that
    score or display the protocol always carry the hash of the exact bytes
    they read. Raises ValueError on any schema violation — a malformed
    protocol must fail loudly, never score silently.
    """
    p = Path(path) if path is not None else PROTOCOL_JSON_PATH
    with open(p, encoding="utf-8") as fh:
        proto = json.load(fh)

    criteria = proto.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{p}: 'criteria' must be a non-empty list")
    seen_ids: set[str] = set()
    for crit in criteria:
        if not isinstance(crit, dict):
            raise ValueError(f"{p}: criterion is not an object: {crit!r}")
        _validate_criterion(crit)
        if crit["id"] in seen_ids:
            raise ValueError(f"{p}: duplicate criterion id {crit['id']!r}")
        seen_ids.add(crit["id"])

    proto["_sha256"] = file_sha256(p)
    return proto
