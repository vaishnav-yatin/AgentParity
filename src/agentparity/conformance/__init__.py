"""Conformance suite: hold an adapter to every capability it claims."""

from .spec import Behaviour, Fixture, Probe
from .suite import CheckResult, Report, Status, Suite, run_suite

__all__ = ["Behaviour", "CheckResult", "Fixture", "Probe", "Report", "Status", "Suite", "run_suite"]
