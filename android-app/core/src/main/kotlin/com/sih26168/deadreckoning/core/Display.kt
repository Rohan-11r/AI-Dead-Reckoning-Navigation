package com.sih26168.deadreckoning.core

import java.util.Locale
import kotlin.math.roundToInt

/**
 * HUD text, kept here so it is unit-tested. "Not available" is always shown as an em dash,
 * never as 0 -- a missing value and a zero value are different facts.
 */
object Display {
    const val NA = "—"

    /** m/s -> "54 km/h" (whole km/h), or NA. */
    fun speedKmh(mps: Double?): String =
        if (mps == null || !mps.isFinite()) NA else "${(mps * 3.6).roundToInt()} km/h"

    /** 0..1 -> "87 %", or NA. */
    fun percent(p: Double?): String =
        if (p == null || !p.isFinite()) NA else "${(p.coerceIn(0.0, 1.0) * 100).roundToInt()} %"

    /** The navigation state as shown to the driver; a simulated outage is always labelled. */
    fun stateLabel(state: GnssState?, simulatedOutage: Boolean): String {
        val s = state?.displayName ?: "STARTING"
        return if (simulatedOutage) "$s (SIMULATED OUTAGE)" else s
    }

    fun roadName(name: String?): String = if (name.isNullOrBlank()) "$NA (map matching not on device yet)" else name

    fun metres(m: Double?): String =
        if (m == null || !m.isFinite()) NA else String.format(Locale.ROOT, "%.1f m", m)

    fun vec(v: Vec3?, digits: Int = 3): String =
        if (v == null) NA else String.format(Locale.ROOT, "%+.${digits}f  %+.${digits}f  %+.${digits}f", v.x, v.y, v.z)

    fun hz(x: Double?): String = if (x == null || !x.isFinite()) NA else String.format(Locale.ROOT, "%.1f Hz", x)

    fun ms(x: Double?): String = if (x == null || !x.isFinite()) NA else String.format(Locale.ROOT, "%.3f ms", x)
}
