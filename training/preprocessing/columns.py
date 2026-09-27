"""Column roles in the synced dataset -- and the mechanical guards that enforce them.

Two project rules are enforced HERE, not by convention:

1. **CAN / VBOX data is ground truth only** (user decision 2026-09-27; SIH26168 forbids
   OBD-II / external speed input). Every vehicle-log column is prefixed ``gt_`` and can
   never appear in ``INPUT_COLUMNS``; ``assert_model_inputs`` rejects it.
2. **No coordinate targets** (AGENTS.md 2.1). Latitude/longitude columns may be used for
   evaluation, never as a regression target; ``assert_model_targets`` rejects them.
"""

from __future__ import annotations

from collections.abc import Iterable

META_COLUMNS = [
    "session_id", "segment_id", "t_utc", "t_utc_phone_raw", "dt_s",
    "alignment_verified", "gt_valid",
]

# Phone-only channels: what the Android device itself can observe.
IMU_COLUMNS = [
    "acc_x_mps2", "acc_y_mps2", "acc_z_mps2",
    "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
    "grav_x_mps2", "grav_y_mps2", "grav_z_mps2",
]
ORIENTATION_COLUMNS = ["orient_azimuth_deg", "orient_pitch_deg", "orient_roll_deg"]
PHONE_GNSS_COLUMNS = [
    "ph_gnss_lat_deg", "ph_gnss_lon_deg", "ph_gnss_alt_m", "ph_gnss_speed_mps",
    "ph_gnss_accuracy_m", "ph_gnss_bearing_deg", "ph_gnss_sats_used", "ph_gnss_sats_visible",
    "ph_gnss_new_fix", "ph_gnss_epoch_utc", "ph_gnss_fix_age_s",
]
INPUT_COLUMNS = IMU_COLUMNS + ORIENTATION_COLUMNS + PHONE_GNSS_COLUMNS

# Ground truth from the V-file (VBOX GNSS + CAN). Labels and evaluation ONLY.
GT_COLUMNS = [
    "gt_lat_deg", "gt_lon_deg", "gt_height_msl_m", "gt_speed_mps", "gt_heading_deg",
    "gt_vvel_mps", "gt_sats", "gt_wheel_speed_mps", "gt_can_speed_mps", "gt_yaw_rate_radps",
    "gt_steering_deg", "gt_long_accel_mps2", "gt_lat_accel_mps2", "gt_brake_on",
    "gt_stationary",
]

# Coordinates: never a model target (AGENTS.md 2.1). ph_gnss_* coordinates may be a
# filter MEASUREMENT (a phone sensor), gt_* coordinates only an evaluation reference.
COORDINATE_COLUMNS = {"gt_lat_deg", "gt_lon_deg", "ph_gnss_lat_deg", "ph_gnss_lon_deg"}

# Substrings that identify vehicle-bus / external-speed channels under ANY name.
_FORBIDDEN_INPUT_MARKERS = ("gt_", "can_", "wheel", "steering", "yaw_rate", "vbox_", "brake",
                            "gear", "engine", "pedal", "clutch", "handbrake")


class ColumnRoleError(ValueError):
    """A column was used in a role the project rules forbid."""


def assert_model_inputs(cols: Iterable[str]) -> list[str]:
    """Raise if any requested model/filter input is ground truth or vehicle-bus data."""
    cols = list(cols)
    bad = [c for c in cols if any(m in c.lower() for m in _FORBIDDEN_INPUT_MARKERS)]
    unknown = [c for c in cols if c not in INPUT_COLUMNS and c not in bad]
    if bad:
        raise ColumnRoleError(
            f"ground-truth / vehicle-bus columns cannot be model or filter inputs: {bad} "
            "(SIH26168 forbids OBD-II / external speed input)")
    if unknown:
        raise ColumnRoleError(f"not a declared phone input column: {unknown}")
    return cols


def assert_model_targets(cols: Iterable[str]) -> list[str]:
    """Raise if a regression target is a coordinate (AGENTS.md 2.1)."""
    cols = list(cols)
    bad = [c for c in cols if c in COORDINATE_COLUMNS or "lat_deg" in c or "lon_deg" in c]
    if bad:
        raise ColumnRoleError(f"latitude/longitude may never be a model target: {bad}")
    return cols
