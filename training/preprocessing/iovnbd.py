"""IO-VNBD raw CSV loader: schema normalisation, parsing hazards, SI conversion.

Every hazard recorded in PROJECT_STATUS.md Phase 0 section 0.4 is handled here explicitly,
plus the ones Phase 2 found on the full dataset. Nothing is imputed or dropped silently:
every parse problem is counted and returned in a ``ParseReport`` (AGENTS.md 2.2).

The dataset holds TWO different file schemas, not one:

* ``S-*.csv`` -- smartphone log, 24 columns (GNSS, accel, gravity, gyro, mag, orientation).
* ``V-*.csv`` -- vehicle log, 29 columns (VBOX GNSS + CAN bus: wheel speeds, steering,
  yaw rate, gear, pedals). Row-synchronised with the S file of the same session.

Encoding
--------
Files are read as cp1252, which is total over every byte present (the body is pure
ASCII). The HEADER line, however, is mixed: ``°`` and ``μ`` are stored as UTF-8 byte pairs
while ``²`` is a single cp1252 byte, so neither codec alone decodes it correctly. Header
tokens are decoded as cp1252 and then individually repaired (``Â°`` -> ``°``).

Units
-----
Conversion to SI happens here, at the I/O boundary (AGENTS.md 3). A channel whose raw
unit label has NOT been verified against the data keeps the ``_raw`` suffix and its
as-labelled unit in ``COLUMN_UNITS``; the Phase 2 audit reports the evidence.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Single constants source (Phase 3). STANDARD_GRAVITY_MPS2 is used here ONLY to convert
# the CAN "g" channels -- never as the navigation gravity model.
from navcore.common.constants import DEG_TO_RAD, KMH_TO_MPS, STANDARD_GRAVITY_MPS2

RAW_ENCODING = "cp1252"
PSI_TO_PA = 6894.757293168
PHONE_TIMEZONE = "Europe/London"  # phone DATE is local civil time; verified vs VBOX UTC
DATE_FORMAT = "%Y-%m-%d %H:%M:%S:%f"  # colon before milliseconds: 2019-09-07 09:13:29:506

SATS_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


class SchemaError(ValueError):
    """A file's header cannot be mapped onto the canonical schema."""


# --------------------------------------------------------------------------------------
# Header normalisation
# --------------------------------------------------------------------------------------

def repair_header_token(token: str) -> str:
    """Undo UTF-8-read-as-cp1252 mojibake in one header token, if and only if it applies.

    ``'GPS ORIENTATION (Â°)'`` -> ``'GPS ORIENTATION (°)'``. A token that is genuinely
    cp1252 (``'m/s²'``) fails the round trip and is returned unchanged.
    """
    try:
        return token.encode(RAW_ENCODING).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return token


def header_key(token: str) -> str:
    """Canonical comparison key for a raw header token.

    Repairs mojibake, applies NFKC (which also folds ``²`` to ``2`` and the micro sign
    U+00B5 to Greek mu U+03BC), transliterates ``°`` -> ``deg`` and ``μ`` -> ``u`` so keys
    are pure ASCII, lowercases, collapses whitespace, and removes whitespace just inside
    parentheses. ``'ORIENTATION (Roll ) (°)'`` -> ``'orientation (roll) (deg)'``;
    ``'ACCELEROMETER X (m/s²) '`` -> ``'accelerometer x (m/s2)'``.
    """
    s = unicodedata.normalize("NFKC", repair_header_token(token))
    s = s.replace("°", "deg").replace("μ", "u")
    s = " ".join(s.lower().split())
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s


# Canonical column -> accepted header keys. Order == file column order.
# Gyro/orientation: the Uncategorised copies label the channels X/Y/Z + Azimuth, the
# Categorised copies relabel the SAME bytes Yaw/Pitch/Roll. The mapping is positional
# (Yaw == X, Pitch == Y, Roll == Z) and is verified physically by the Phase 2 audit
# against the vehicle CAN yaw rate -- see reports/phase2/dataset_report.json.
PHONE_SCHEMA: list[tuple[str, tuple[str, ...]]] = [
    ("gps_lat_deg", ("gps latitude (degrees)",)),
    ("gps_lon_deg", ("gps longitude (degrees)",)),
    ("gps_alt_m", ("gps altitude (m)",)),
    # Header says "Kmh" but the values are m/s (Android Location.getSpeed); Phase 3 measured
    # logged / VBOX[m/s] = 0.997 over 59 sessions. Canonical name records the raw value.
    ("gps_speed_raw", ("gps speed (kmh)",)),
    ("gps_accuracy_m", ("gps accuracy (m)",)),
    ("gps_bearing_deg", ("gps orientation (deg)",)),
    ("gps_sats_raw", ("gps satellites in range",)),
    ("t_rel_ms", ("time since start (ms)",)),
    # Uncategorised S-Vfa01/S-Vfa02 omit the closing paren -- an explicit, known variant.
    ("date_raw", ("date (yyyy-mo-dd hh-mi-ss_sss)", "date (yyyy-mo-dd hh-mi-ss_sss")),
    ("acc_x_mps2", ("accelerometer x (m/s2)",)),
    ("acc_y_mps2", ("accelerometer y (m/s2)",)),
    ("acc_z_mps2", ("accelerometer z (m/s2)",)),
    ("grav_x_mps2", ("gravity x (m/s2)",)),
    ("grav_y_mps2", ("gravity y (m/s2)",)),
    ("grav_z_mps2", ("gravity z (m/s2)",)),
    ("gyro_x_radps", ("gyroscope x (rad/s)", "gyroscope yaw (rad/s)")),
    ("gyro_y_radps", ("gyroscope y (rad/s)", "gyroscope pitch (rad/s)")),
    ("gyro_z_radps", ("gyroscope z (rad/s)", "gyroscope roll (rad/s)")),
    ("mag_x_uT", ("magnetic field x (ut)",)),
    ("mag_y_uT", ("magnetic field y (ut)",)),
    ("mag_z_uT", ("magnetic field z (ut)",)),
    ("orient_azimuth_deg", ("orientation (azimuth) (deg)", "orientation (yaw) (deg)")),
    ("orient_pitch_deg", ("orientation (pitch) (deg)",)),
    ("orient_roll_deg", ("orientation (roll) (deg)",)),
]

VEHICLE_SCHEMA: list[tuple[str, tuple[str, ...]]] = [
    ("vbox_sats", ("no of gps satellites available",)),
    ("t_utc_tod_s", ("time since start of day (seconds)",)),
    ("vbox_lat_deg", ("latitude (degrees)",)),
    ("vbox_lon_deg", ("longitude (degrees)",)),
    ("vbox_speed_kmh", ("velocity (km/hr)",)),
    ("vbox_heading_deg", ("heading (degrees)",)),
    ("vbox_height_raw", ("height (km)",)),
    ("vbox_vvel_raw", ("vertical velocity (km/hr)",)),
    ("sample_period_s", ("sample period (seconds)",)),
    ("steering_angle_deg", ("steering angle (degrees)",)),
    ("wheel_speed_fl_raw", ("wheel speed front left (rad/sec)",)),
    ("wheel_speed_fr_raw", ("wheel speed front right (rad/sec)",)),
    ("wheel_speed_rl_raw", ("wheel speed rear left (rad/sec)",)),
    ("wheel_speed_rr_raw", ("wheel speed rear right (rad/sec)",)),
    ("yaw_rate_degps", ("yaw rate (deg/sec)",)),
    ("can_speed_kmh", ("indicated vehicle speed (km/hr)",)),
    ("can_long_accel_g", ("indicated longitudinal acceleration (g)",)),
    ("can_lat_accel_g", ("indicated lateral acceleration (g)",)),
    ("handbrake", ("handbrake (0 or 1)",)),
    ("gear_requested", ("gear requested (number fof gear employed 1-5)",)),
    ("gear", ("gear (number fof gear employed 1-5)",)),
    ("engine_rpm", ("engine speed (rev/min)",)),
    ("coolant_temp_c", ("coolant temperature (degrees)",)),
    ("clutch", ("clutch position (0 or 1)",)),
    ("brake_pressure_psi", ("brake pressure (psi)",)),
    ("brake_on", ("brake position (0 or 1)",)),
    ("battery_v", ("battery voltage (volts)",)),
    ("air_temp_c", ("air temperature (degrees)",)),
    ("accel_pedal_raw", ("accelerator pedal position (0 or 1)",)),
]

SCHEMAS = {"S": PHONE_SCHEMA, "V": VEHICLE_SCHEMA}

# Units of the columns in the PROCESSED (SI) frames. "_raw" columns carry their label unit,
# explicitly marked unverified. Serialised into every parquet file's metadata.
COLUMN_UNITS: dict[str, str] = {
    # phone
    "row_idx": "1", "t_rel_s": "s", "t_local": "local civil time (Europe/London)",
    "t_utc": "UTC", "gps_lat_deg": "deg (WGS84)", "gps_lon_deg": "deg (WGS84)",
    "gps_alt_m": "m", "gps_speed_mps": "m/s", "gps_accuracy_m": "m",
    "gps_bearing_deg": "deg", "gps_sats_used": "count", "gps_sats_visible": "count",
    "gps_sats_excel_recovered": "bool (value repaired from Excel date damage)",
    "acc_x_mps2": "m/s^2", "acc_y_mps2": "m/s^2", "acc_z_mps2": "m/s^2",
    "grav_x_mps2": "m/s^2", "grav_y_mps2": "m/s^2", "grav_z_mps2": "m/s^2",
    "gyro_x_radps": "rad/s", "gyro_y_radps": "rad/s", "gyro_z_radps": "rad/s",
    "mag_x_uT": "uT", "mag_y_uT": "uT", "mag_z_uT": "uT",
    "orient_azimuth_deg": "deg", "orient_pitch_deg": "deg", "orient_roll_deg": "deg",
    # vehicle
    "vbox_sats": "count", "t_utc_tod_s": "s since 00:00 UTC", "vbox_lat_deg": "deg (WGS84)",
    "vbox_lon_deg": "deg (WGS84)", "vbox_speed_mps": "m/s", "vbox_heading_deg": "deg",
    "vbox_height_raw": "labelled km -- UNVERIFIED, see audit", "vbox_vvel_raw":
    "labelled km/h -- UNVERIFIED, see audit", "sample_period_s": "s",
    "steering_angle_deg": "deg", "wheel_speed_fl_raw": "labelled rad/s -- UNVERIFIED",
    "wheel_speed_fr_raw": "labelled rad/s -- UNVERIFIED",
    "wheel_speed_rl_raw": "labelled rad/s -- UNVERIFIED",
    "wheel_speed_rr_raw": "labelled rad/s -- UNVERIFIED", "yaw_rate_radps": "rad/s",
    "can_speed_mps": "m/s", "can_long_accel_mps2": "m/s^2 (label g x 9.80665)",
    "can_lat_accel_mps2": "m/s^2 (label g x 9.80665)", "handbrake": "0/1",
    "gear_requested": "gear number", "gear": "gear number", "engine_rpm": "rev/min",
    "coolant_temp_c": "degC", "clutch": "0/1", "brake_pressure_pa": "Pa", "brake_on": "0/1",
    "battery_v": "V", "air_temp_c": "degC",
    "accel_pedal_raw": "labelled '0 or 1' -- UNVERIFIED, see audit",
}


def map_header(raw_header: str, kind: str) -> tuple[list[str], dict[str, str]]:
    """Map a raw header line onto canonical column names.

    Returns ``(canonical_names, {canonical: raw_token})``. Raises ``SchemaError`` on a
    column count mismatch or any unrecognised token -- never guesses.
    """
    schema = SCHEMAS[kind]
    tokens = raw_header.rstrip("\r\n").split(",")
    if len(tokens) != len(schema):
        raise SchemaError(f"{kind}-file header has {len(tokens)} columns, expected {len(schema)}")
    names: list[str] = []
    raw_by_name: dict[str, str] = {}
    for pos, (tok, (canon, accepted)) in enumerate(zip(tokens, schema, strict=True)):
        key = header_key(tok)
        if key not in accepted:
            raise SchemaError(
                f"{kind}-file column {pos}: header {tok!r} (key {key!r}) is not one of {accepted}"
            )
        names.append(canon)
        raw_by_name[canon] = tok
    return names, raw_by_name


# --------------------------------------------------------------------------------------
# Field parsers
# --------------------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
_MON = "|".join(_MONTHS)
EXCEL_MON_YY_RE = re.compile(rf"^\s*({_MON})-(\d{{1,2}})\s*$")  # '12 / 14' -> 'Dec-14'
EXCEL_DD_MON_RE = re.compile(rf"^\s*(\d{{1,2}})-({_MON})\s*$")  # '10 / 12' -> '10-Dec'


def parse_satellites(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """Split ``'18 / 19'`` into (used, visible) nullable integers.

    The first number is taken as satellites USED in the fix and the second as satellites
    VISIBLE ("in range"); the audit asserts used <= visible on every row, which is the
    invariant that ordering implies.

    Excel damage is repaired, not dropped: the CSVs were at some point opened in Excel,
    which turned ``'12 / 14'`` into the date ``'Dec-14'`` (month/2-digit-year, used when
    the d/m reading is invalid) and ``'10 / 12'`` into ``'10-Dec'`` (UK day/month). Both
    transforms are deterministic and lossless, so the integers are recovered exactly and
    the rows are flagged in the returned boolean series.

    Returns ``(used, visible, excel_recovered, malformed)``. Blanks become <NA>; any
    other unparseable string becomes <NA> and is counted in ``malformed``.
    """
    s = raw.astype("string").str.strip()
    parts = s.str.extract(SATS_RE)
    used = pd.to_numeric(parts[0], errors="coerce").astype("Int16")
    visible = pd.to_numeric(parts[1], errors="coerce").astype("Int16")
    recovered = pd.Series(False, index=s.index)

    mon_yy = s.str.extract(EXCEL_MON_YY_RE)
    hit = mon_yy[0].notna()
    if hit.any():
        used[hit] = mon_yy.loc[hit, 0].map(_MONTHS).astype("Int16")
        visible[hit] = pd.to_numeric(mon_yy.loc[hit, 1]).astype("Int16")
        recovered |= hit
    dd_mon = s.str.extract(EXCEL_DD_MON_RE)
    hit = dd_mon[0].notna()
    if hit.any():
        used[hit] = pd.to_numeric(dd_mon.loc[hit, 0]).astype("Int16")
        visible[hit] = dd_mon.loc[hit, 1].map(_MONTHS).astype("Int16")
        recovered |= hit

    malformed = int((s.notna() & (s != "") & used.isna()).sum())
    return used, visible, recovered, malformed


def parse_phone_datetime(raw: pd.Series) -> tuple[pd.Series, int]:
    """Parse ``'2019-09-07 09:13:29:506'`` (colon before ms) with an explicit format.

    Auto-inference is deliberately not used: it misreads the trailing ``:506``. Returns
    naive local timestamps and the count of non-blank values that failed to parse.
    """
    s = raw.astype("string").str.strip()
    parsed = pd.to_datetime(s, format=DATE_FORMAT, errors="coerce")
    failures = int((s.notna() & (s != "") & parsed.isna()).sum())
    return parsed.astype("datetime64[ms]"), failures


def local_to_utc(local: pd.Series, tz: str = PHONE_TIMEZONE) -> tuple[pd.Series, int]:
    """Localise naive civil time to ``tz`` and convert to UTC.

    Times that are ambiguous or non-existent at a DST transition become NaT and are
    counted rather than guessed.
    """
    utc = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    lost = int((local.notna() & utc.isna()).sum())
    return utc, lost


# --------------------------------------------------------------------------------------
# File loading
# --------------------------------------------------------------------------------------

@dataclass
class ParseReport:
    """Everything that went wrong (or was odd) while parsing one file. Nothing hidden."""

    path: str
    kind: str
    n_rows: int = 0
    header_raw: dict[str, str] = field(default_factory=dict)
    header_variant: str = ""
    coerced_non_numeric: dict[str, int] = field(default_factory=dict)
    sats_malformed: int = 0
    sats_excel_recovered: int = 0
    date_unparsed: int = 0
    utc_lost_at_dst: int = 0


def read_header(path: Path) -> str:
    with open(path, "rb") as fh:
        return fh.readline().decode(RAW_ENCODING)


def _read_numeric_csv(path: Path, names: list[str], str_cols: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read with the fast C parser. Numeric columns are float64; if a numeric column
    contains non-numeric text the file is re-read as strings and the offending cells are
    coerced to NaN and COUNTED per column."""
    dtypes = {n: ("string" if n in str_cols else "float64") for n in names}
    common = dict(
        encoding=RAW_ENCODING, header=0, names=names, skipinitialspace=True,
        na_values=[""], keep_default_na=False, engine="c",
    )
    try:
        return pd.read_csv(path, dtype=dtypes, **common), {}
    except ValueError:
        df = pd.read_csv(path, dtype="string", **common)
        coerced: dict[str, int] = {}
        for n in names:
            if n in str_cols:
                continue
            num = pd.to_numeric(df[n], errors="coerce")
            bad = int((df[n].notna() & num.isna()).sum())
            if bad:
                coerced[n] = bad
            df[n] = num.astype("float64")
        return df, coerced


def load_raw(path: Path, kind: str) -> tuple[pd.DataFrame, ParseReport]:
    """Load one raw file into its canonical-name, RAW-unit frame (no conversion yet)."""
    header = read_header(path)
    names, raw_by_name = map_header(header, kind)
    rep = ParseReport(path=str(path), kind=kind, header_raw=raw_by_name)
    if kind == "S":
        rep.header_variant = "XYZ/Azimuth" if "gyroscope x" in header_key(header) else "Yaw/Pitch/Roll"
        str_cols = {"gps_sats_raw", "date_raw"}
    else:
        rep.header_variant = "vehicle"
        str_cols = set()
    df, rep.coerced_non_numeric = _read_numeric_csv(path, names, str_cols)
    rep.n_rows = len(df)
    return df, rep


def to_si_phone(raw: pd.DataFrame, rep: ParseReport) -> pd.DataFrame:
    """Canonical SI phone frame. Converts km/h and ms; splits satellites; parses time."""
    out = pd.DataFrame(index=raw.index)
    out["row_idx"] = np.arange(len(raw), dtype=np.int32)
    out["t_rel_s"] = raw["t_rel_ms"] / 1000.0
    t_local, rep.date_unparsed = parse_phone_datetime(raw["date_raw"])
    out["t_local"] = t_local
    out["t_utc"], rep.utc_lost_at_dst = local_to_utc(t_local)
    for c in ("gps_lat_deg", "gps_lon_deg", "gps_alt_m"):
        out[c] = raw[c]
    # NOT converted: the "Kmh" label is wrong, the logged value is already m/s
    # (reports/phase3/phone_gps_speed_unit.json). Phase 0-2 divided by 3.6 -- a bug.
    out["gps_speed_mps"] = raw["gps_speed_raw"]
    out["gps_accuracy_m"] = raw["gps_accuracy_m"]
    out["gps_bearing_deg"] = raw["gps_bearing_deg"]
    used, visible, recovered, rep.sats_malformed = parse_satellites(raw["gps_sats_raw"])
    out["gps_sats_used"] = used
    out["gps_sats_visible"] = visible
    out["gps_sats_excel_recovered"] = recovered
    rep.sats_excel_recovered = int(recovered.sum())
    for c in raw.columns:
        if c.startswith(("acc_", "grav_", "gyro_", "mag_", "orient_")):
            out[c] = raw[c]
    return out


def to_si_vehicle(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical SI vehicle frame. Only VERIFIED unit labels are converted."""
    out = pd.DataFrame(index=raw.index)
    out["row_idx"] = np.arange(len(raw), dtype=np.int32)
    out["vbox_sats"] = raw["vbox_sats"].round().astype("Int16")
    for c in ("t_utc_tod_s", "vbox_lat_deg", "vbox_lon_deg"):
        out[c] = raw[c]
    out["vbox_speed_mps"] = raw["vbox_speed_kmh"] * KMH_TO_MPS
    out["vbox_heading_deg"] = raw["vbox_heading_deg"]
    out["vbox_height_raw"] = raw["vbox_height_raw"]
    out["vbox_vvel_raw"] = raw["vbox_vvel_raw"]
    out["sample_period_s"] = raw["sample_period_s"]
    out["steering_angle_deg"] = raw["steering_angle_deg"]
    for w in ("fl", "fr", "rl", "rr"):
        out[f"wheel_speed_{w}_raw"] = raw[f"wheel_speed_{w}_raw"]
    out["yaw_rate_radps"] = raw["yaw_rate_degps"] * DEG_TO_RAD
    out["can_speed_mps"] = raw["can_speed_kmh"] * KMH_TO_MPS
    out["can_long_accel_mps2"] = raw["can_long_accel_g"] * STANDARD_GRAVITY_MPS2
    out["can_lat_accel_mps2"] = raw["can_lat_accel_g"] * STANDARD_GRAVITY_MPS2
    for c in ("handbrake", "gear_requested", "gear", "engine_rpm", "coolant_temp_c", "clutch"):
        out[c] = raw[c]
    out["brake_pressure_pa"] = raw["brake_pressure_psi"] * PSI_TO_PA
    for c in ("brake_on", "battery_v", "air_temp_c", "accel_pedal_raw"):
        out[c] = raw[c]
    return out


# --------------------------------------------------------------------------------------
# Hashing for deduplication
# --------------------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def content_digest(raw: pd.DataFrame, decimals: int = 9) -> str:
    """Hash of the parsed VALUES, independent of header labels and float formatting.

    Needed because the Categorised and Uncategorised copies of each smartphone file
    differ byte-wise only by float serialisation (``1.539`` vs ``1.5390000000000001``) and
    header labels. Numeric columns are rounded to ``decimals`` places (the raw data carries
    at most 7), NaN preserved; string columns hashed verbatim after stripping.
    """
    h = hashlib.sha256()
    h.update(",".join(raw.columns).encode())
    for c in raw.columns:
        col = raw[c]
        if pd.api.types.is_float_dtype(col):
            h.update(np.round(col.to_numpy(dtype=np.float64), decimals).tobytes())
        else:
            h.update("\x1f".join(col.fillna("<NA>").astype(str).str.strip()).encode())
    return h.hexdigest()


def max_abs_numeric_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Largest absolute difference between two same-shape canonical frames (NaN == NaN)."""
    worst = 0.0
    for c in a.columns:
        if pd.api.types.is_float_dtype(a[c]):
            d = np.abs(a[c].to_numpy() - b[c].to_numpy())
            both_nan = np.isnan(a[c].to_numpy()) & np.isnan(b[c].to_numpy())
            one_nan = np.isnan(a[c].to_numpy()) ^ np.isnan(b[c].to_numpy())
            if one_nan.any():
                return float("inf")
            d = d[~both_nan]
            if d.size:
                worst = max(worst, float(d.max()))
    return worst


# --------------------------------------------------------------------------------------
# Path metadata
# --------------------------------------------------------------------------------------

CATEGORY_DIR_RE = re.compile(r"^(?P<cat>\w+) \(Driver (?P<driver>[A-Z])\)$")
FILE_RE = re.compile(r"^(?P<kind>[SV])-(?P<sid>[A-Za-z0-9]+)\.csv$", re.IGNORECASE)
SEGMENT_SUFFIX_RE = re.compile(r"^(?P<base>.*\d)(?P<seg>[a-z])$")


@dataclass(frozen=True)
class RawFileInfo:
    path: Path
    rel: str
    collection: str  # "categorised" | "uncategorised"
    kind: str  # "S" | "V"
    session_key: str  # lowercase stem without prefix: "vta1a"
    stem: str  # original-case stem without prefix: "Vta1a" / "vta2"
    category: str | None  # Categorised only: "Vta", "S", "M", ...
    driver: str | None  # Categorised only: "A".."E"


def describe_raw_file(path: Path, root: Path) -> RawFileInfo:
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    m = FILE_RE.match(parts[-1])
    if m is None:
        raise SchemaError(f"unrecognised file name {rel!r}")
    top = parts[0].lower()
    collection = "categorised" if top.startswith("categorised") else "uncategorised"
    category = driver = None
    if collection == "categorised":
        cm = CATEGORY_DIR_RE.match(parts[1])
        if cm is None:
            raise SchemaError(f"cannot read category/driver from {parts[1]!r}")
        category, driver = cm["cat"], cm["driver"]
    return RawFileInfo(
        path=path, rel=rel, collection=collection, kind=m["kind"].upper(),
        session_key=m["sid"].lower(), stem=m["sid"], category=category, driver=driver,
    )


def route_group(session_id: str) -> str:
    """Sessions split into lettered segments of one drive (S3a/S3b/S3c, Vw14a/b/c) belong
    to one route group and must never be separated across train/val/test."""
    m = SEGMENT_SUFFIX_RE.match(session_id)
    return m["base"] if m else session_id
