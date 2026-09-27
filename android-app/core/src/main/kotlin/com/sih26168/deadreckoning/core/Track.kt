package com.sih26168.deadreckoning.core

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * DISPLAY ONLY: recent positions as local East/North metres around the first one, for the
 * track canvas. Navigation never uses this (its geodesy is the audited path in the Python
 * reference / edge engine); it only turns positions the engine already produced into
 * screen-friendly metres. Equirectangular about the anchor with the WGS84 meridian and
 * prime-vertical radii (the same formulas as navcore.geometry.geodesy); error < 0.1 % within
 * ~10 km of the anchor -- far below a pixel on the track view.
 */
class TrackProjector(private val maxPoints: Int = 3000) {
    private var lat0 = Double.NaN
    private var lon0 = Double.NaN
    private var rm = 0.0
    private var rnCos = 0.0
    private val pts = ArrayDeque<Pair<Double, Double>>()

    val points: List<Pair<Double, Double>> get() = pts.toList()

    fun add(latRad: Double, lonRad: Double) {
        if (lat0.isNaN()) {
            lat0 = latRad
            lon0 = lonRad
            rm = meridianRadius(latRad)
            rnCos = primeVerticalRadius(latRad) * cos(latRad)
        }
        pts.addLast(((lonRad - lon0) * rnCos) to ((latRad - lat0) * rm))
        while (pts.size > maxPoints) pts.removeFirst()
    }

    fun clear() {
        pts.clear()
        lat0 = Double.NaN
    }

    companion object {
        const val WGS84_A = 6378137.0
        const val WGS84_F = 1.0 / 298.257223563
        const val WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

        /** Meridian radius of curvature [m] (navcore.geometry.geodesy.meridian_radius). */
        fun meridianRadius(latRad: Double): Double {
            val d = sqrt(1.0 - WGS84_E2 * sin(latRad) * sin(latRad))
            return WGS84_A * (1.0 - WGS84_E2) / (d * d * d)
        }

        /** Prime-vertical radius of curvature [m] (navcore.geometry.geodesy.prime_vertical_radius). */
        fun primeVerticalRadius(latRad: Double): Double = WGS84_A / sqrt(1.0 - WGS84_E2 * sin(latRad) * sin(latRad))
    }
}
