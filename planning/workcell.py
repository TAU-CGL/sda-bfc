"""Workcell geometry: the floor and optional walls, as halfspaces.

The floor is always the xy-plane at the height of the lower robot base;
walls are vertical planes parallel to {x=0} or {y=0}.  The free side of a
wall is the side containing R1's base (the origin).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Wall:
    axis: str      # "x" or "y": the plane {axis = offset}
    offset: float

    def halfspace(self, margin=0.0):
        direction = np.array([1.0, 0.0, 0.0]) if self.axis == "x" \
            else np.array([0.0, 1.0, 0.0])
        sign = 1.0 if self.offset >= 0.0 else -1.0
        return sign * direction, abs(self.offset) - margin


@dataclass
class Workcell:
    floor_z: float = 0.0
    walls: list = field(default_factory=list)

    def halfspaces(self, translation_margin=(0.0, 0.0, 0.0),
                   tilt_margins=(0.0, 0.0, 0.0)):
        """(normal, offset) pairs; margins shrink the free region to cover
        the querying arm's own placement uncertainty.  tilt_margins are the
        rotation angles that can displace points along each axis (yaw never
        moves z, roll never moves x, pitch never moves y), already scaled by
        the caller's lever arms."""
        spaces = [(np.array([0.0, 0.0, -1.0]),
                   -self.floor_z - translation_margin[2] - tilt_margins[2])]
        for wall in self.walls:
            normal, offset = wall.halfspace()
            axis = 0 if abs(normal[0]) > 0.5 else 1
            spaces.append((normal, offset - translation_margin[axis]
                           - tilt_margins[axis]))
        return spaces


def floor_below_both_robots(x_placement):
    return min(0.0, float(x_placement[2, 3]))


def workcell_for_placement(x_placement, walls=()):
    return Workcell(floor_z=floor_below_both_robots(x_placement),
                    walls=list(walls))
