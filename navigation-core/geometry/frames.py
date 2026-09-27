"""Explicit reference-frame management (AGENTS.md 3: "no implicit frame changes").

Frames (docs/navigation_math.md section 1):

=========  ======  ==================================================================
Frame      symbol  definition
=========  ======  ==================================================================
BODY       b       The phone IMU / device frame. Android convention: x to the right of
                   the screen, y to its top, z out of the screen. ``DEVICE`` is an alias.
VEHICLE    v       x forward, y left, z up (FLU). Related to ``b`` by the mounting
                   rotation, which is ESTIMATED from data (Phase 4) -- never assumed.
ENU        n       Local tangent navigation frame: East, North, Up. The navigation frame.
NED        ned     North, East, Down. Supported for interchange; not used internally.
ECEF       e       Earth-centred, Earth-fixed.
GEODETIC   --      WGS84 (lat, lon, h). Not a vector frame: reached only through
                   geometry.geodesy (the single audited path to coordinates).
=========  ======  ==================================================================

A ``Rotation`` is tagged ``dst <- src`` and stores a unit quaternion ``q_dst_src``
(= ``R_src^dst``). Applying it to a ``FramedVector`` in any other frame, or composing two
rotations whose frames do not chain, raises ``FrameMismatchError``.

IO-VNBD caveat (Phase 2 finding): the logged gyro COLUMNS are not in the accelerometer's
axis order -- the vertical rate appears in column 2. Mapping raw columns into ``BODY`` is
a calibration output (Phase 4), not something this module guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from navcore.geometry.geodesy import R_ENU_TO_NED, R_ecef_to_enu
from navcore.geometry.quaternion import (
    Q_IDENTITY,
    dcm_to_q,
    is_unit,
    q_conj,
    q_mul,
    q_normalize,
    q_rotate,
    q_to_dcm,
)


class Frame(Enum):
    BODY = "b"
    DEVICE = "b"  # alias of BODY: the phone IMU is the body
    VEHICLE = "v"
    ENU = "n"
    NED = "ned"
    ECEF = "e"
    GEODETIC = "geo"


class FrameMismatchError(ValueError):
    """An operation was asked to combine quantities expressed in different frames."""


def _require_vector_frame(f: Frame) -> None:
    if f is Frame.GEODETIC:
        raise FrameMismatchError("GEODETIC is not a vector frame; use geometry.geodesy")


@dataclass(frozen=True)
class FramedVector:
    """A free 3-vector (or stack ``(..., 3)``) tagged with the frame it is expressed in."""

    v: np.ndarray
    frame: Frame

    def __post_init__(self) -> None:
        v = np.asarray(self.v, dtype=np.float64)
        if v.shape[-1] != 3:
            raise ValueError(f"FramedVector needs last dimension 3, got {v.shape}")
        _require_vector_frame(self.frame)
        v = v.copy()
        v.setflags(write=False)
        object.__setattr__(self, "v", v)

    def __add__(self, other: FramedVector) -> FramedVector:
        if not isinstance(other, FramedVector):
            return NotImplemented
        if other.frame is not self.frame:
            raise FrameMismatchError(f"cannot add {other.frame.name} to {self.frame.name}")
        return FramedVector(self.v + other.v, self.frame)

    def __sub__(self, other: FramedVector) -> FramedVector:
        if not isinstance(other, FramedVector):
            return NotImplemented
        if other.frame is not self.frame:
            raise FrameMismatchError(f"cannot subtract {other.frame.name} from {self.frame.name}")
        return FramedVector(self.v - other.v, self.frame)


@dataclass(frozen=True)
class Rotation:
    """Rotation ``dst <- src``: ``x_dst = R_src^dst x_src``, stored as ``q_dst_src``."""

    q: np.ndarray
    src: Frame
    dst: Frame

    def __post_init__(self) -> None:
        _require_vector_frame(self.src)
        _require_vector_frame(self.dst)
        q = np.asarray(self.q, dtype=np.float64)
        if q.shape[-1] != 4:
            raise ValueError(f"Rotation quaternion must have last dimension 4, got {q.shape}")
        if not is_unit(q, tol=1e-6):
            raise ValueError("Rotation quaternion must be unit-norm (normalise explicitly)")
        q = q_normalize(q)
        q.setflags(write=False)
        object.__setattr__(self, "q", q)

    @classmethod
    def identity(cls, frame: Frame) -> Rotation:
        return cls(Q_IDENTITY, frame, frame)

    @classmethod
    def from_matrix(cls, R, src: Frame, dst: Frame) -> Rotation:
        return cls(dcm_to_q(R), src, dst)

    def as_matrix(self) -> np.ndarray:
        return q_to_dcm(self.q)

    def inverse(self) -> Rotation:
        return Rotation(q_conj(self.q), self.dst, self.src)

    def apply(self, vec: FramedVector) -> FramedVector:
        if not isinstance(vec, FramedVector):
            raise TypeError("Rotation.apply takes a FramedVector; tag raw arrays with a frame")
        if vec.frame is not self.src:
            raise FrameMismatchError(
                f"rotation {self.dst.name}<-{self.src.name} applied to a vector in {vec.frame.name}")
        return FramedVector(q_rotate(self.q, vec.v), self.dst)

    def __matmul__(self, other: Rotation) -> Rotation:
        """(c <- b) @ (b <- a) = (c <- a)."""
        if not isinstance(other, Rotation):
            return NotImplemented
        if other.dst is not self.src:
            raise FrameMismatchError(
                f"cannot compose {self.dst.name}<-{self.src.name} with "
                f"{other.dst.name}<-{other.src.name}")
        return Rotation(q_mul(self.q, other.q), other.src, self.dst)


# ---- Standard transforms ---------------------------------------------------------------

def enu_from_ecef(lat_rad: float, lon_rad: float) -> Rotation:
    """R_e^n at the given origin (section 3.3). Rotates vectors; positions also need the
    origin offset -- use geometry.geodesy.LocalTangentPlane for those."""
    return Rotation.from_matrix(R_ecef_to_enu(lat_rad, lon_rad), Frame.ECEF, Frame.ENU)


def ned_from_enu() -> Rotation:
    return Rotation.from_matrix(R_ENU_TO_NED, Frame.ENU, Frame.NED)


def nav_from_body(q_nb) -> Rotation:
    """Attitude: q_nb represents R_b^n (section 4.1)."""
    return Rotation(q_nb, Frame.BODY, Frame.ENU)


def vehicle_from_body(q_vb) -> Rotation:
    """Mounting rotation R_b^v. Supplied by calibration (Phase 4); never assumed."""
    return Rotation(q_vb, Frame.BODY, Frame.VEHICLE)
