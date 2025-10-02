"""Orientation helpers derived from RLBot physics rotations."""
from __future__ import annotations

import math
from dataclasses import dataclass

from rlbot.utils.structures.game_data_struct import Rotator

from vec import Vec3


@dataclass(frozen=True)
class Orientation:
    """Forward/right/up vectors derived from a ``Rotator``."""

    forward: Vec3
    right: Vec3
    up: Vec3

    @classmethod
    def from_rotator(cls, rotation: Rotator) -> "Orientation":
        pitch = float(rotation.pitch)
        yaw = float(rotation.yaw)
        roll = float(rotation.roll)

        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cr = math.cos(roll)
        sr = math.sin(roll)

        forward = Vec3(cp * cy, cp * sy, sp)
        right = Vec3(sy * sp * sr + cr * cy, -cy * sp * sr + cr * sy, -cp * sr)
        up = Vec3(-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr)
        return cls(forward=forward, right=right, up=up)


def relative_location(origin: Vec3, orientation: Orientation, target: Vec3) -> Vec3:
    """Return ``target`` relative to ``origin`` in car coordinates."""

    offset = target - origin
    return Vec3(
        offset.dot(orientation.forward),
        offset.dot(orientation.right),
        offset.dot(orientation.up),
    )
