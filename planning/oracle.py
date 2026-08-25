"""The collision oracle boundary.

A CollisionOracle is the only component that knows where R2 truly stands.
The simulation oracles implement it with the ground-truth placement; a real
robot implements it with force sensing.  Everything above this interface
(belief, planning, session) is oracle-agnostic by construction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

STATIC_ARM = "A"
DYNAMIC_ARM = "B"


@dataclass(frozen=True)
class MoveOutcome:
    reached: bool
    contact: tuple | None   # (link_a, link_b) or ("env", name); None if reached


@dataclass(frozen=True)
class TouchReport:
    pair: tuple
    interior: bool


class CollisionOracle(ABC):
    """Guarded motion against the true world; stops at the first contact."""

    @abstractmethod
    def move(self, arm, target) -> MoveOutcome:
        ...

    @abstractmethod
    def configuration(self, arm):
        ...

    @abstractmethod
    def classify_touch(self) -> TouchReport | None:
        """Describe the current contact, if any."""
        ...
