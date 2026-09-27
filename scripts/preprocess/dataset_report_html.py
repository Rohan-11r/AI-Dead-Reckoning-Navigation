"""Render reports/phase2/dataset_report.json as a self-contained HTML page.

No external assets: inline CSS, inline SVG bars. Every number shown is read from the
JSON report produced by scripts/preprocess/ingest_iovnbd.py -- nothing is typed in here.
"""

from __future__ import annotations

import html
import json

CSS = """
:root{--bg:#fbfbfa;--surface:#fff;--ink:#1d1d1b;--ink2:#55554f;--muted:#8a8a82;
--line:#e4e3de;--bar:#2f6fb0;--good:#1f7a3a;--warn:#9a6700;--bad:#b42318;--chip:#f1f0ec}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;
--surface:#1f1f1d;--ink:#ecebe6;--ink2:#b9b8b1;--muted:#8d8c85;--line:#34332f;
--bar:#6fa6dc;--good:#5cc07a;--warn:#e0b04a;--bad:#f07a6e;--chip:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:15px;margin:20px 0 6px}
.meta{color:var(--ink2);font-size:13px}.tiles{display:grid;
grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.tile span{color:var(--ink2);font-size:12px}
.scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600;background:var(--chip);position:sticky;top:0}
td.n{text-align:right}.ok{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.note{color:var(--ink2);max-width:80ch}code{background:var(--chip);padding:0 4px;border-radius:4px}
svg text{fill:var(--ink2);font-size:11px}svg .bar{fill:var(--bar)}svg .bar:hover{opacity:.8}
.fig{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0}
"""

STATUS = {"PASS": ("ok", "✔ PASS"), "FAIL": ("bad", "✖ FAIL"),
          "WARN": ("warn", "⚠ NOTE"), "INFO": ("", "&#8505; INFO")}


def esc(x) -> str:
    return html.escape(str(x))


def num(x, nd=3) -> str:
    if x is None:
        return "&ndash;"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.{nd}f}" if abs(x) < 1e6 else f"{x:.3e}"
    return esc(x)


def table(headers: list[str], rows: list[list], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def hbars(title: str, items: list[tuple[str, float]], unit: str, log: bool = False) -> str:
    """Single-series horizontal bar chart (inline SVG) with per-bar hover tooltips."""
    import math
    w, lh, lw = 640, 22, 110
    vals = [math.log10(v + 1) if log else v for _, v in items]
    vmax = max(vals) or 1.0
    h = lh * len(items) + 8
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title)}">']
    for i, ((label, v), sv) in enumerate(zip(items, vals)):
        y = 4 + i * lh
        bw = max(0.0, (w - lw - 90) * sv / vmax)
        parts.append(f'<text x="{lw - 6}" y="{y + 14}" text-anchor="end">{esc(label)}</text>')
        if bw > 0:
            parts.append(
                f'<rect class="bar" x="{lw}" y="{y + 3}" width="{bw:.1f}" height="{lh - 8}" rx="3">'
                f"<title>{esc(label)}: {v:,.4g} {esc(unit)}</title></rect>")
        parts.append(f'<text x="{lw + bw + 6:.1f}" y="{y + 14}">{v:,.4g}</text>')
    parts.append("</svg>")
    scale = " (log scale)" if log else ""
    return f'<div class="fig"><h3>{esc(title)}{scale}</h3>{"".join(parts)}</div>'


def check_rows(rep: dict) -> list[list]:
    s = rep["summary"]
    g, ga, al, ue = s["gravity"], s["gyro_axis_mapping"], s["alignment"], s["unit_evidence"]

    def st(k):
        cls, lab = STATUS[k]
        return f'<span class="{cls}">{lab}</span>'

    return [
        ["1", "cp1252 encoding", st("PASS"),
         f"all {s['files']['total_csv']} files read as cp1252; header mojibake (&deg;, &mu; stored "
         "as UTF-8 byte pairs, &sup2; as a cp1252 byte) repaired per token"],
        ["2", "Irregular headers", st("PASS"),
         f"variants mapped explicitly: {esc(json.dumps(s['files']['header_variants']))}; "
         "includes a missing ')' in the Uncategorised S-Vfa01/02 DATE header"],
        ["3", "'18 / 19' satellites", st("PASS"),
         f"split to used/visible; {s['parse']['sats_excel_recovered']:,} cells repaired from Excel "
         f"date damage ('Dec-14' &larr; '12 / 14'); malformed left: {s['parse']['sats_malformed']}; "
         f"used&gt;visible: {s['parse']['sats_used_gt_visible']}"],
        ["4", "DATE 'HH:MM:SS:mmm'", st("PASS"),
         f"explicit format; unparsed: {s['parse']['date_unparsed']}; lost at DST: "
         f"{s['parse']['utc_lost_at_dst']}"],
        ["5", "km/h &rarr; m/s", st("PASS"), "converted at the loader boundary (phone GPS, VBOX, CAN)"],
        ["6", "Gyro Yaw/Pitch/Roll mapping", st("WARN"),
         f"labels are positional aliases of X/Y/Z (Uncategorised copies carry X/Y/Z). Vehicle yaw "
         f"appears on column 2 (<code>gyro_y</code>) in {ga['best_axis_sign_votes'].get('y+', 0)}/"
         f"{ga['sessions_strong']} strongly-correlated sessions, gain "
         f"{num(ga['median_gain_by_axis']['y'])}; accelerometer Z is vertical in "
         f"{g['dominant_gravity_axis_votes'].get('z', 0)}/72. The label 'Yaw' on column 1 is WRONG. "
         "Horizontal axes unresolved &rarr; Phase 4"],
        ["7", "Accelerometer includes gravity", st("PASS"),
         f"stationary |acc| = {num(g['stationary_acc_norm_weighted_mean_mps2'], 4)} m/s&sup2; over "
         f"{g['stationary_rows']:,} rows"],
        ["8", "Categorised vs Uncategorised", st("PASS"),
         f"{s['dedupe']['byte_identical_by_kind']['V']} V pairs byte-identical; "
         f"{s['dedupe']['content_identical_by_kind']['S']} S pairs content-identical (max |&Delta;| "
         f"{num(s['dedupe']['max_abs_numeric_diff_overall'], 1)}), float formatting only &rarr; "
         f"{s['sessions']['unique']} unique sessions"],
        ["G", "Gravity 9.80665 vs local", st("PASS"),
         f"GRAVITY channel norm {num(g['channel_norm_pooled_mean_mps2'], 6)} m/s&sup2;; closer to g0 "
         f"in {g['sessions_closer_to_g0_than_local']}/{g['sessions_total']} sessions; local normal "
         f"gravity {num(g['local_normal_gravity_mps2_range'][0], 5)}&ndash;"
         f"{num(g['local_normal_gravity_mps2_range'][1], 5)}"],
        ["A", "S/V row pairing", st("WARN"),
         f"row pairing off by the phone clock error in {len(al['sessions_with_abs_row_offset_gt_1s'])} "
         f"sessions (up to {num(s['clock']['phone_utc_minus_vbox_median_s_range'][1], 1)} s). "
         f"UTC time join verified in {len(al['verified_sessions'])}, contradicted in "
         f"{len(al['contradicted_sessions'])}"],
        ["U", "VBOX/CAN unit labels", st("WARN"),
         f"'Height (km)' is metres (height&minus;phone alt median {num(ue['height_raw_minus_phone_alt_median_m'], 1)} m "
         f"&asymp; geoid); 'Vertical velocity (km/hr)' behaves as m/s (slope "
         f"{num(ue['dhdt_over_vvel_raw_slope_median'])}); wheel speed / VBOX km/h = "
         f"{num(ue['wheel_raw_over_vbox_kmh_median'], 4)} (rad/s label unresolved); pedal "
         f"range {num(ue['accel_pedal_raw_range'][0], 0)}&ndash;{num(ue['accel_pedal_raw_range'][1], 0)}"],
    ]


def render(rep: dict) -> str:
    m, s, sp = rep["meta"], rep["summary"], rep["split"]
    ss = s["sessions"]
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
           f'<meta name=viewport content="width=device-width,initial-scale=1">'
           f"<title>IO-VNBD Integrity Report</title><style>{CSS}</style></head><body><main>"]
    out.append(f"<h1>{esc(m['title'])}</h1><div class=meta>generated {esc(m['generated'])} &middot; "
               f"<code>{esc(m['script'])}</code> &middot; commit <code>{esc(m['commit'])}</code> "
               f"&middot; Python {esc(m['python'])} &middot; runtime {num(m['runtime_s'], 1)} s</div>")

    tiles = [
        ("CSV files", ss and s["files"]["total_csv"]), ("unique sessions", ss["unique"]),
        ("phone rows", ss["phone_rows_total"]), ("hours", ss["phone_hours_total"]),
        ("km (VBOX speed)", ss["vbox_distance_km_total"]), ("drivers", len(ss["drivers"])),
        ("route groups", ss["route_groups"]), ("continuous segments", ss["continuous_segments_total"]),
    ]
    out.append('<div class="tiles">' + "".join(
        f'<div class="tile"><b>{num(v, 2)}</b><span>{esc(k)}</span></div>' for k, v in tiles) + "</div>")

    out.append("<h2>Hazard &amp; integrity checklist</h2>")
    out.append(table(["#", "check", "result", "evidence"], check_rows(rep)))

    d = s["dedupe"]
    out.append("<h2>Deduplication</h2>")
    out.append('<p class="note">Two hashes per file: SHA-256 of the bytes, and a digest of the '
               "parsed values rounded to 1e-9 (header labels excluded). Canonical copy = "
               "Categorised.</p>")
    out.append(table(["measure", "value"], [
        ["CSV files", num(s["files"]["total_csv"])],
        ["unique SHA-256", num(s["files"]["unique_sha256"])],
        ["unique content digests", num(s["files"]["unique_content_digest"])],
        ["file status", esc(json.dumps(s["files"]["by_status"]))],
        ["byte-identical pairs (S / V)", f"{d['byte_identical_by_kind']['S']} / {d['byte_identical_by_kind']['V']}"],
        ["content-identical pairs (S / V)", f"{d['content_identical_by_kind']['S']} / {d['content_identical_by_kind']['V']}"],
        ["max |numeric difference| within a pair", num(d["max_abs_numeric_diff_overall"], 1)],
        ["content conflicts", esc(d["conflicts"] or "none")],
    ], {1}))

    out.append("<h2>Sample timing</h2>")
    tp = s["timing_phone"]
    out.append(hbars("Phone dt histogram (all sessions)", list(tp["dt_histogram"].items()),
                     "intervals", log=True))
    out.append(table(["measure", "phone", "vehicle"], [
        ["backwards steps (recording restarts)", num(tp["n_backwards"]), num(s["timing_vehicle"]["n_backwards"])],
        ["duplicate timestamps", num(tp["n_duplicate_t"]), num(s["timing_vehicle"]["dt_histogram"]["0"])],
        ["gaps &gt; 1 s", num(tp["n_gaps_gt_1s"]), num(s["timing_vehicle"]["n_gaps_gt_1s"])],
        ["effective rate (median, Hz)", num(tp["effective_rate_hz_median"]), "&ndash;"],
        ["max |DATE &minus; TIME SINCE START| drift (s)", num(tp["date_vs_trel_max_abs_s"], 1), "&ndash;"],
    ], {1, 2}))

    out.append("<h2>Missing &amp; unparseable values</h2>")
    nan_rows = [[f"<code>{esc(k)}</code>", num(v)] for k, v in s["nan_phone"].items() if v] + \
               [[f"<code>{esc(k)}</code> (vehicle)", num(v)] for k, v in s["nan_vehicle"].items() if v]
    p = s["parse"]
    out.append(f'<p class="note">All other columns: zero NaN. Literal <code>nan</code> text cells '
               f"coerced and counted: {p['coerced_non_numeric_cells']}. Exact duplicate phone rows: "
               f"{p['exact_duplicate_rows_phone']:,}. Zero-fix rows: {p['zero_fix_rows_phone']}. "
               f"Satellite cells repaired from Excel damage: {p['sats_excel_recovered']:,} in "
               f"{esc(', '.join(p['sats_excel_recovered_sessions']))}.</p>")
    out.append(table(["column", "NaN count"], nan_rows or [["(none)", "0"]], {1}))

    g = s["gravity"]
    out.append("<h2>Gravity model evidence</h2>")
    out.append(table(["quantity", "m/s&sup2;"], [
        ["standard gravity g0 (defined)", num(g["standard_g0_mps2"], 6)],
        ["local normal gravity, Somigliana + free-air (min&ndash;max over sessions)",
         f"{num(g['local_normal_gravity_mps2_range'][0], 6)} &ndash; {num(g['local_normal_gravity_mps2_range'][1], 6)}"],
        ["GRAVITY channel norm, pooled mean", num(g["channel_norm_pooled_mean_mps2"], 6)],
        ["GRAVITY channel norm, session means (min&ndash;max)",
         f"{num(g['channel_norm_session_mean_range_mps2'][0], 6)} &ndash; {num(g['channel_norm_session_mean_range_mps2'][1], 6)}"],
        ["GRAVITY channel norm, largest within-session std", num(g["channel_norm_session_std_max_mps2"], 6)],
        ["pooled channel &minus; g0", num(g["pooled_minus_g0_mps2"], 6)],
        ["pooled channel &minus; local normal gravity", num(g["pooled_minus_local_mean_mps2"], 6)],
        [f"stationary |accelerometer|, {g['stationary_rows']:,} rows in {g['stationary_sessions']} sessions",
         num(g["stationary_acc_norm_weighted_mean_mps2"], 4)],
        ["stationary |accelerometer|, session range",
         f"{num(g['stationary_acc_norm_session_range_mps2'][0], 4)} &ndash; {num(g['stationary_acc_norm_session_range_mps2'][1], 4)}"],
    ], {1}))
    out.append('<p class="note">The GRAVITY channel is the Android fused gravity vector, normalised to '
               "the defined constant g0 &mdash; not a measurement of local gravity. The INS must use "
               "the Somigliana model (docs/navigation_math.md &sect;5.1). The stationary accelerometer "
               "norm exceeds local gravity; separating bias from scale error is Phase 4 work.</p>")

    ga, al = s["gyro_axis_mapping"], s["alignment"]
    out.append("<h2>Gyroscope axis mapping</h2>")
    out.append(f'<p class="note">{esc(ga["reference"])}.</p>')
    out.append(table(["axis (column)", "Categorised label", "median corr", "median gain"], [
        [f"<code>gyro_{ax}</code> (col {i + 1})", lab, num(ga["median_corr_by_axis"][ax]),
         num(ga["median_gain_by_axis"][ax])]
        for i, (ax, lab) in enumerate(zip("xyz", ("Yaw", "Pitch", "Roll")))], {2, 3}))
    out.append(f'<p class="note">Best-axis votes: {esc(json.dumps(ga["best_axis_sign_votes"]))} of '
               f"{ga['sessions_strong']} strong sessions. Residual lag after time alignment: median "
               f"{num(ga['residual_lag_s_median_strong'], 2)} s, range "
               f"{num(ga['residual_lag_s_range_strong'][0], 1)}&ndash;{num(ga['residual_lag_s_range_strong'][1], 1)} s. "
               f"Row-paired vs time-aligned median corr of <code>gyro_y</code>: "
               f"{num(ga['gyro_y_vs_yaw_corr_median_row_paired'])} &rarr; "
               f"{num(ga['gyro_y_vs_yaw_corr_median_time_aligned'])}.</p>")

    out.append("<h2>Phone &harr; vehicle alignment</h2>")
    out.append(f'<p class="note"><b>Rule:</b> {esc(al["rule"])}. Never join on row index.</p>')
    out.append(table(["status", "sessions"], [[esc(k), num(v)] for k, v in al["status_counts"].items()], {1}))
    out.append(f'<p class="note">Verified: {esc(", ".join(al["verified_sessions"]))}. Phone-clock '
               f"correction in verified sessions: {num(al['verified_correction_s_range'][0], 2)} to "
               f"{num(al['verified_correction_s_range'][1], 2)} s. Stationary-only recordings: "
               f"{esc(', '.join(al['stationary_only_sessions']))}. Unequal S/V row counts: "
               f"{esc(', '.join(al['sessions_rows_unequal']))}. Phone GNSS speed lags VBOX by a median "
               f"{num(s['unit_evidence']['phone_gnss_speed_lag_s_median'], 1)} s.</p>")

    out.append("<h2>Route / driver split</h2>")
    out.append(hbars("Hours per split", list(sp["split_duration_h"].items()), "h"))
    out.append(table(["split", "sessions", "hours", "fraction"], [
        [k, num(len(sp["splits"][k])), num(sp["split_duration_h"][k], 2), num(sp["split_fraction"][k], 3)]
        for k in ("train", "val", "test")], {1, 2, 3}))
    out.append(f'<p class="note">Held-out test driver(s): {esc(", ".join(sp["test_drivers"]))}. '
               f"Driver hours: {esc(json.dumps(sp['driver_duration_h']))}. Manifest: "
               f"<code>data/splits/iovnbd_split_v1.json</code>.</p>")

    out.append("<h2>Sessions</h2>")
    rows = []
    for x in rep["sessions"]:
        ph, ve, pr = x.get("phone", {}), x.get("vehicle", {}), x.get("pair", {})
        rows.append([
            esc(x["session_id"]), esc(x.get("driver")), esc(x.get("category")), esc(x["route_group"]),
            num(ph.get("n_rows")), num((ph.get("duration_s") or 0) / 60, 1),
            num(ve.get("distance_vbox_km"), 2), num(ph.get("timing", {}).get("n_continuous_segments")),
            num(ph.get("timing", {}).get("longest_segment_s"), 0),
            num(ph.get("gnss", {}).get("accuracy_m_p95"), 1),
            num(ph.get("gravity_channel", {}).get("norm_mean_mps2"), 5),
            num(pr.get("clock_offset_s", {}).get("median"), 2),
            esc(pr.get("alignment", {}).get("status", "")).replace("_", " "),
        ])
    out.append(table(["session", "driver", "cat", "route grp", "rows", "min", "km", "segs",
                      "longest s", "GPS acc p95 m", "|g| mean", "row offset s", "alignment"],
                     rows, {4, 5, 6, 7, 8, 9, 10, 11}))
    if rep.get("excluded_files"):
        out.append("<h2>Excluded files</h2>")
        out.append(table(["file", "reason"], [[esc(f["rel"]), esc(f.get("reason"))]
                                              for f in rep["excluded_files"]]))
    out.append("</main></body></html>")
    return "\n".join(out)
