# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-imposed wall-clock budget for a step's SEQUENTIAL wait stack.

Why a step needs one
--------------------
The orchestrator enforces its per-step ``timeout:`` with
``subprocess.run(timeout=...)``, which SIGKILLs the child. No signal is
delivered, so no ``except`` / ``finally`` compensating cleanup runs, and a
result payload that is printed once at the very end is never emitted. For a
step that CREATES cloud resources that combination is a leak with no
reclamation path: teardown receives no ownership provenance and — correctly —
refuses to delete a resource it cannot prove this run created.

A step whose waits run in SEQUENCE therefore must not depend on the config cap
having been sized against arithmetic in a comment. Each wait carries its own
independent timeout, the sum drifts every time one of them is retuned, and the
comment is the last thing to be updated. Instead the step stamps a budget at
entry and derives every wait that follows from what is LEFT of it. The provider
config then has to clear ONE number (the budget plus the floors below) instead
of tracking every internal wait, and the kill can only ever arrive after the
step has already cleaned up and printed its payload.

Nothing here shortens a wait that fits: every accessor returns the wait's own
full value while the budget can still fund it, so a healthy run behaves exactly
as it did before the budget existed. The clamp engages only on the tail, where
the alternative is the SIGKILL.

Floors are deliberate. A wait given no time at all fails a fixture that may be
perfectly healthy, so each derived wait keeps a short window. Floors are also
the ONLY way a step can exceed its budget, which is what makes the enforced
bound computable and quotable in the provider config:

    bound = total + sum(floors granted after the budget expires)
"""

from __future__ import annotations

import time

from common.ssh_utils import SSH_PROBE_TIMEOUT_S


class StepBudget:
    """Wall clock a step allows itself, stamped at entry.

    Stamp this BEFORE the first cloud call so every derived deadline accounts
    for work already done, then wrap each subsequent wait in
    :meth:`wait_timeout` (timeout-shaped waits) or :meth:`probe_attempts`
    (SSH probe ladders).
    """

    def __init__(self, total: float) -> None:
        self.total = float(total)
        self._started = time.monotonic()

    def elapsed(self) -> float:
        """Seconds since the budget was stamped."""
        return time.monotonic() - self._started

    def remaining(self) -> float:
        """Seconds still available; negative once the budget is spent."""
        return self.total - self.elapsed()

    def wait_timeout(self, full: int, *, floor: int) -> int:
        """Clamp one timeout-shaped wait to what is left of the budget.

        ``full`` is the wait's own worst-case timeout and is returned unchanged
        whenever it still fits. ``floor`` is the shortest window the wait is
        still granted once the budget is gone.
        """
        return max(floor, min(full, int(self.remaining())))

    def probe_attempts(self, full_attempts: int, *, interval: int, floor: int = 1) -> int:
        """Clamp an SSH probe ladder to what is left of the budget.

        Against an unreachable guest each attempt costs its ``interval`` PLUS
        the probe's own connect timeout (``ssh_utils.SSH_PROBE_TIMEOUT_S``), so
        ``attempts * (interval + SSH_PROBE_TIMEOUT_S)`` is the ladder's real
        worst case and the number the remaining budget has to divide.
        """
        per_attempt = max(1, interval + SSH_PROBE_TIMEOUT_S)
        affordable = int(self.remaining() // per_attempt)
        return max(floor, min(full_attempts, affordable))

    def can_afford(self, seconds: float) -> bool:
        """True while ``seconds`` of further work still fits inside the budget.

        Used to stop an ITERATIVE envelope (e.g. a multi-zone create walk)
        before it starts an attempt the budget cannot fund, which is the only
        way to bound a loop whose per-iteration wait is already clamped.
        """
        return self.remaining() >= seconds
