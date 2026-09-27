"""Sensor sample containers: IMU, GNSS -- validated, timestamped, frame-tagged.

Every sample carries:
* ``t_s``   -- time in seconds on ONE monotonic session clock (float64). Samples never
  carry a nominal rate; consumers derive the real ``dt`` from consecutive ``t_s``.
* its frame, explicitly (``Frame``): an IMU vector is meaningless without it.

Units are SI and in the field names. Construction validates shape, finiteness and
physical plausibility and RAISES -- a bad sample is never silently patched
(AGENTS.md 5 "no silent fallbacks").

GNSS latency: a phone reports a fix some time after the instant it describes
(IO-VNBD: ~4.1 s). ``GnssSample.t_s`` is the EPOCH the fix describes, and
``t_received_s`` when it became available; filters must never use a fix before
``t_received_s`` (causality) and must compare it with the state at ``t_s``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from navcore.geometry.frames import Frame

# Plausibility limits for a road vehicle + phone MEMS IMU. Values outside are sensor or
# parsing faults, not dynamics (smartphone accelerometers saturate at 8-16 g).
MAX_SPECIFIC_FORCE_MPS2 = 16 * 9.80665
MAX_ANGULAR_RATE_RADPS = 35.0  # ~2000 deg/s, the usual MEMS gyro full scale
MAX_GROUND_SPEED_MPS = 100.0


class SensorSampleError(ValueError):
    """A sample violates its schema or physical plausibility."""


def _vec3(name: str, v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (3,):
        raise SensorSampleError(f"{name} must have shape (3,), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise SensorSampleError(f"{name} contains non-finite values: {a}")
    a = a.copy()
    a.setflags(write=False)
    return a


def _time(t_s: float) -> float:
    t = float(t_s)
    if not math.isfinite(t):
        raise SensorSampleError(f"timestamp must be finite, got {t_s!r}")
    return t


@dataclass(frozen=True)
class SensorSample:
    """Base: a timestamp on the session clock."""

    t_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_s", _time(self.t_s))


@dataclass(frozen=True)
class ImuSample(SensorSample):
    """Specific force [m/s^2] and angular rate [rad/s], both expressed in ``frame``.

    ``gyro_valid`` marks which gyro axes carry a usable measurement (IO-VNBD: only the
    vertical one -- reports/phase4/gyro_axis_resolution.json). Invalid axes are 0 and
    must not be treated as information.
    """

    accel_mps2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_radps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: Frame = Frame.BODY
    gyro_valid: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        super().__post_init__()
        a, w = _vec3("accel_mps2", self.accel_mps2), _vec3("gyro_radps", self.gyro_radps)
        if np.linalg.norm(a) > MAX_SPECIFIC_FORCE_MPS2:
            raise SensorSampleError(f"specific force {np.linalg.norm(a):.1f} m/s^2 exceeds full scale")
        if np.max(np.abs(w)) > MAX_ANGULAR_RATE_RADPS:
            raise SensorSampleError(f"angular rate {np.max(np.abs(w)):.1f} rad/s exceeds full scale")
        if self.frame in (Frame.GEODETIC, Frame.ECEF):
            raise SensorSampleError(f"IMU samples are body-fixed, not {self.frame.name}")
        valid = tuple(bool(x) for x in self.gyro_valid)
        if len(valid) != 3:
            raise SensorSampleError("gyro_valid must have 3 entries")
        if any(not ok and w[i] != 0.0 for i, ok in enumerate(valid)):
            raise SensorSampleError("invalid gyro axes must be zero, not a guessed value")
        object.__setattr__(self, "accel_mps2", a)
        object.__setattr__(self, "gyro_radps", w)
        object.__setattr__(self, "gyro_valid", valid)


@dataclass(frozen=True)
class GnssSample(SensorSample):
    """A GNSS fix. ``t_s`` = epoch it describes; ``t_received_s`` = when it was available.

    Position is WGS84 geodetic in RADIANS / ellipsoidal metres. Horizontal velocity, when
    present, is ground speed [m/s] + course over ground (bearing, radians clockwise from
    North, the GNSS convention); ``velocity_enu()`` converts it once, here.
    """

    lat_rad: float = 0.0
    lon_rad: float = 0.0
    h_m: float = 0.0
    horizontal_accuracy_m: float = float("nan")
    t_received_s: float | None = None
    speed_mps: float | None = None
    bearing_rad: float | None = None
    sats_used: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for n in ("lat_rad", "lon_rad", "h_m"):
            object.__setattr__(self, n, _time(getattr(self, n)) if n != "h_m" else float(getattr(self, n)))
        if not abs(self.lat_rad) <= math.pi / 2:
            raise SensorSampleError(f"latitude {self.lat_rad} rad outside [-pi/2, pi/2] (degrees passed?)")
        if not abs(self.lon_rad) <= math.pi:
            raise SensorSampleError(f"longitude {self.lon_rad} rad outside [-pi, pi] (degrees passed?)")
        if not math.isfinite(self.h_m) or abs(self.h_m) > 20_000:
            raise SensorSampleError(f"implausible height {self.h_m} m")
        acc = float(self.horizontal_accuracy_m)
        if not (math.isfinite(acc) and acc > 0):
            raise SensorSampleError(f"horizontal_accuracy_m must be finite and > 0, got {acc}")
        object.__setattr__(self, "horizontal_accuracy_m", acc)
        rec = self.t_s if self.t_received_s is None else _time(self.t_received_s)
        if rec < self.t_s:
            raise SensorSampleError("a fix cannot be received before the epoch it describes")
        object.__setattr__(self, "t_received_s", rec)
        if (self.speed_mps is None) != (self.bearing_rad is None):
            raise SensorSampleError("speed and bearing come together or not at all")
        if self.speed_mps is not None:
            s, b = float(self.speed_mps), float(self.bearing_rad)
            if not (math.isfinite(s) and 0 <= s <= MAX_GROUND_SPEED_MPS and math.isfinite(b)):
                raise SensorSampleError(f"implausible speed/bearing {s}, {b}")
            object.__setattr__(self, "speed_mps", s)
            object.__setattr__(self, "bearing_rad", b)

    @property
    def latency_s(self) -> float:
        return float(self.t_received_s) - self.t_s

    def velocity_enu(self) -> np.ndarray | None:
        """Horizontal velocity [E, N] in m/s from speed + course (clockwise from North)."""
        if self.speed_mps is None:
            return None
        return np.array([self.speed_mps * math.sin(self.bearing_rad),
                         self.speed_mps * math.cos(self.bearing_rad)])


class TimestampGuard:
    """Real-``dt`` bookkeeping for a sample stream. Returns dt for accepted samples and
    COUNTS (never hides) backwards steps, duplicates and gaps."""

    def __init__(self, max_gap_s: float = 1.0) -> None:
        self.max_gap_s = float(max_gap_s)
        self.last_t: float | None = None
        self.n_accepted = self.n_backwards = self.n_duplicates = self.n_gaps = 0

    def step(self, t_s: float) -> float | None:
        """dt [s] since the previous accepted sample; None = reject (or first sample, or a
        gap that must restart integration -- check ``n_gaps``)."""
        t = _time(t_s)
        if self.last_t is None:
            self.last_t = t
            self.n_accepted += 1
            return None
        dt = t - self.last_t
        if dt < 0:
            self.n_backwards += 1
            return None
        if dt == 0:
            self.n_duplicates += 1
            return None
        self.last_t = t
        self.n_accepted += 1
        if dt > self.max_gap_s:
            self.n_gaps += 1
            return None
        return dt


@dataclass(frozen=True)
class ImuAxisMap:
    """Raw logged columns -> BODY axes: ``body = M @ raw``, ``M`` a signed permutation,
    with a per-axis validity mask. Constructed from EVIDENCE, never assumed."""

    accel: np.ndarray
    gyro: np.ndarray
    gyro_valid: tuple[bool, bool, bool]
    evidence: str

    def __post_init__(self) -> None:
        for n in ("accel", "gyro"):
            M = np.asarray(getattr(self, n), dtype=np.float64)
            if M.shape != (3, 3) or not np.all(np.isin(M, (-1.0, 0.0, 1.0))):
                raise SensorSampleError(f"{n} map must be a 3x3 signed (partial) permutation")
            if np.any(np.sum(np.abs(M), axis=1) > 1) or np.any(np.sum(np.abs(M), axis=0) > 1):
                raise SensorSampleError(f"{n} map uses a raw column twice or fills an axis twice")
            M = M.copy()
            M.setflags(write=False)
            object.__setattr__(self, n, M)
        for i, ok in enumerate(self.gyro_valid):
            if ok != bool(np.any(self.gyro[i])):
                raise SensorSampleError("gyro_valid must mark exactly the mapped axes")

    def to_body(self, t_s: float, accel_raw, gyro_raw) -> ImuSample:
        return ImuSample(t_s=t_s, accel_mps2=self.accel @ np.asarray(accel_raw, dtype=np.float64),
                         gyro_radps=self.gyro @ np.asarray(gyro_raw, dtype=np.float64),
                         frame=Frame.BODY, gyro_valid=self.gyro_valid)


# IO-VNBD (Android phone, logged columns [c1, c2, c3]).
# Accelerometer: Android device axes as logged (gravity on +z when flat: 72/72 sessions).
# Gyro: column 2 = device z (Phase 2: sign +, gain 0.997 vs CAN yaw rate). Columns 1/3 are
# NOT usable as angular rates in any signed permutation (Phase 4 stop-to-stop test,
# reports/phase4/gyro_axis_resolution.json) -> mapped to nothing and flagged invalid.
IOVNBD_AXIS_MAP = ImuAxisMap(
    accel=np.eye(3),
    gyro=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    gyro_valid=(False, False, True),
    evidence="reports/phase4/gyro_axis_resolution.json; reports/phase2/dataset_report.json",
)
