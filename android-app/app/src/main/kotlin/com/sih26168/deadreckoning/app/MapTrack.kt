package com.sih26168.deadreckoning.app

import com.sih26168.deadreckoning.core.GnssState

/**
 * One engine position for the map, in degrees exactly as the engine reported it, plus whether
 * the engine was dead reckoning (no GNSS fused: simulated outage or state LOST) when it
 * produced it. Display only: nothing here feeds back into the engine.
 */
data class GeoTrackPoint(val latDeg: Double, val lonDeg: Double, val deadReckoning: Boolean)

/**
 * Splits [points] into runs of equal [GeoTrackPoint.deadReckoning], in order. Every run after
 * the first starts with the previous run's last point, so the drawn line is continuous across
 * a GNSS <-> dead-reckoning change.
 */
fun trackRuns(points: List<GeoTrackPoint>): List<Pair<Boolean, List<GeoTrackPoint>>> {
    val runs = ArrayList<Pair<Boolean, List<GeoTrackPoint>>>()
    var start = 0
    for (i in 1..points.size) {
        if (i == points.size || points[i].deadReckoning != points[start].deadReckoning) {
            runs += points[start].deadReckoning to points.subList(if (start > 0) start - 1 else 0, i)
            start = i
        }
    }
    return runs
}

/**
 * Whether the UI shows "GNSS out" (banner, orange car). The engine reports [GnssState.LOST]
 * before it has initialised (mode WAITING: no position, nothing is being dead-reckoned), so
 * LOST alone does not count; neither does a snapshot left over from a stopped session. With
 * the simulated outage off, this follows the engine: it clears on the first fused fix.
 */
fun gnssOutShown(state: GnssState?, mode: String?, outageOn: Boolean, running: Boolean): Boolean =
    running && (outageOn || (state == GnssState.LOST && mode != MODE_WAITING))

/** The dead-reckoning engine's lifecycle mode before its first moving fix. */
const val MODE_WAITING = "WAITING"
