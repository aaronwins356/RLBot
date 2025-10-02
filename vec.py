"""Lightweight vector math utilities for Rocket League bots."""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt
from typing import Iterable


@dataclass(frozen=True)
class Vec3:
    """Immutable 3D vector with helper operations.

    RLBot exposes positions and velocities through ``Vector3`` objects.  Converting
    them into this simple dataclass makes the rest of the code base agnostic to the
    source structure while still supporting arithmetic operators.
    """

    x: float
    y: float
    z: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vec3":
        x, y, z = values
        return cls(float(x), float(y), float(z))

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> "Vec3":
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    __rmul__ = __mul__

    def __truediv__(self, scale: float) -> "Vec3":
        inv = 1.0 / scale
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        mag = self.magnitude()
        if mag == 0:
            return Vec3(0.0, 0.0, 0.0)
        return self / mag

    def distance(self, other: "Vec3") -> float:
        return (self - other).magnitude()

    def flattened(self) -> "Vec3":
        return Vec3(self.x, self.y, 0.0)

    def angle_between(self, other: "Vec3") -> float:
        denom = max(self.magnitude() * other.magnitude(), 1e-6)
        cos_theta = max(min(self.dot(other) / denom, 1.0), -1.0)
        return acos(cos_theta)

    def clamp_magnitude(self, max_value: float) -> "Vec3":
        mag = self.magnitude()
        if mag <= max_value:
            return self
        return self * (max_value / mag)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __iter__(self):
        yield from (self.x, self.y, self.z)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Vec3(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"
