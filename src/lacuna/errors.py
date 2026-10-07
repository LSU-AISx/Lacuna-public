"""Typed errors exposed by the Lacuna front end."""

from __future__ import annotations

from dataclasses import dataclass


class LacunaError(Exception):
    """Base class for user-facing Lacuna failures."""


class DSLParseError(LacunaError):
    """The model text is not valid Lacuna DSL."""


class ResolutionError(LacunaError):
    """The model is valid DSL but cannot be resolved."""


class CapabilityError(ResolutionError):
    """The model is valid but outside the current implementation slice."""


class PrecisionResolutionError(ResolutionError):
    """Target rounding invalidates a value or a checked mathematical condition."""


@dataclass(frozen=True)
class CoreFailureDiagnostic:
    """Structured context for a bounded C runtime resource failure."""

    resource: str
    capacity: int
    occupancy: int
    peak: int
    event_kind: str | None = None
    event_phase: str | None = None
    event_index: int | None = None
    node: int | None = None
    t: float | None = None


class CoreError(LacunaError):
    """The C evaluator rejected an operation."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        diagnostic: CoreFailureDiagnostic | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.diagnostic = diagnostic
