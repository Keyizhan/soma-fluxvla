"""Hold generated text between updates, never observations or action KV."""
from __future__ import annotations

import math


class SubtaskHold:
    def __init__(self, interval_s: float = 0.0):
        if not math.isfinite(interval_s) or interval_s < 0:
            raise ValueError("subtask update interval must be finite and non-negative")
        self.interval_s = float(interval_s)
        self._record = None
        self._generation_id = 0

    def get(self, instruction: str, now: float):
        record = self._record
        if (record is None or record["instruction"] != instruction
                or not 0 <= now - record["started"] < self.interval_s):
            return None
        return record

    def commit(self, instruction: str, started: float, value: dict):
        self._generation_id += 1
        self._record = {
            "instruction": instruction, "started": started,
            "generation_id": self._generation_id, "value": value,
        }
        return self._record
