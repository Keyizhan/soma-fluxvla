"""Causal text debounce. Holds labels only; never caches observations or action KV."""
from __future__ import annotations

from dataclasses import dataclass
import math


def label_key(text: str) -> tuple[str, ...]:
    """Ignore field punctuation, spacing and parallel-clause ordering only."""
    return tuple(sorted(" ".join(part.lower().replace("_", " ").strip()
                                 .rstrip(".; ").split()) for part in text.split("||")))


@dataclass(frozen=True)
class Decision:
    instruction: str
    observed_at: float
    active: str
    pending: str | None
    count: int
    reason: str
    switched: bool


class SubtaskStability:
    def __init__(self, confirmations: int = 1, reset_after_s: float = 5.0, key_fn=label_key, refresh_same=False):
        if isinstance(confirmations, bool) or not isinstance(confirmations, int) or confirmations < 1:
            raise ValueError("confirmations must be a positive integer")
        if not math.isfinite(reset_after_s) or reset_after_s <= 0:
            raise ValueError("reset_after_s must be finite and positive")
        self.key_fn = key_fn
        self.refresh_same = refresh_same
        self.confirmations = confirmations
        self.reset_after_s = reset_after_s
        self._state: Decision | None = None

    def propose(self, instruction: str, candidate: str, now: float,
                *, fresh: bool = True) -> Decision:
        """Pure proposal: a failed action inference must not advance votes."""
        if not math.isfinite(now) or not any(self.key_fn(candidate)):
            raise ValueError("invalid observation time or empty subtask")
        old = self._state
        if old is None or old.instruction != instruction or not 0 <= now - old.observed_at < self.reset_after_s:
            return Decision(instruction, now, candidate, None, 0, "initialize", False)
        if not fresh:
            return Decision(instruction, old.observed_at, old.active, old.pending,
                            old.count, "reuse_no_vote", False)
        if self.confirmations == 1:
            return Decision(instruction, now, candidate, None, 0, "passthrough",
                            self.key_fn(candidate) != self.key_fn(old.active))
        if self.key_fn(candidate) == self.key_fn(old.active):
            return Decision(instruction, now, candidate if self.refresh_same else old.active, None, 0, "same_task", False)
        count = old.count + 1 if old.pending is not None and self.key_fn(candidate) == self.key_fn(old.pending) else 1
        if count >= self.confirmations:
            return Decision(instruction, now, candidate, None, 0, "confirmed", True)
        return Decision(instruction, now, old.active, candidate, count, "await_confirmation", False)

    def commit(self, decision: Decision) -> None:
        self._state = decision
