package com.sih26168.deadreckoning.core

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Measured sensor rate for the diagnostics screen: events in the last [windowS] seconds of
 * the stream's OWN timestamps (not wall-clock arrival, which the UI thread would distort),
 * plus the median interval and a gap count. Android's requested rate is a hint; this is
 * what actually arrived.
 */
class RateMeter(private val windowS: Double = 1.0, private val gapS: Double = 0.5) {
    private val times = ArrayDeque<Double>()
    var nTotal = 0L
        private set
    var nGaps = 0L
        private set
    private var last: Double? = null

    fun record(tS: Double) {
        val prev = last
        if (prev != null && tS - prev > gapS) nGaps++
        last = tS
        nTotal++
        times.addLast(tS)
        while (times.isNotEmpty() && times.first() < tS - windowS) times.removeFirst()
    }

    /** Events per second over the window (0 until two events are seen). */
    val rateHz: Double
        get() {
            if (times.size < 2) return 0.0
            val span = times.last() - times.first()
            return if (span > 0) (times.size - 1) / span else 0.0
        }

    /** Median interval between consecutive events in the window [s], or null. */
    val medianDtS: Double?
        get() {
            if (times.size < 2) return null
            val d = times.zipWithNext { a, b -> b - a }.sorted()
            return d[d.size / 2]
        }
}

/**
 * The "Simulate GNSS outage" toggle. While [enabled], every fix is WITHHELD from the
 * navigation engine -- the same thing `simulation/outage` does with a `full` event on a
 * recording -- but still handed to the logger flagged `withheld_by_sim`, so a recorded demo
 * drive states exactly where the outage was simulated. Withholding goes by the fix EPOCH,
 * like the reference simulator.
 */
class SimulatedOutage {
    var enabled: Boolean = false
        private set
    private var startS: Double? = null
    val windows = mutableListOf<Pair<Double, Double>>()  // completed [start, end) windows
    var nWithheld = 0L
        private set

    fun set(on: Boolean, nowS: Double) {
        if (on == enabled) return
        enabled = on
        if (on) {
            startS = nowS
        } else {
            windows += (startS ?: nowS) to nowS
            startS = null
        }
    }

    /** True if [fix] must be withheld from the engine. */
    fun withholds(fix: GnssSample): Boolean {
        val s = startS
        val w = enabled && s != null && fix.tS >= s
        if (w) nWithheld++
        return w
    }
}

/**
 * Confidence shown on the HUD: the probability that the true horizontal position lies
 * within [radiusM] of the reported one, for a circular Gaussian with per-axis sigma
 * [sigmaHm]:  P(r < R) = 1 - exp(-R^2 / (2 sigma^2)).
 * A defined, checkable quantity -- not an arbitrary score. Null sigma -> null.
 */
fun positionConfidence(sigmaHm: Double?, radiusM: Double = 10.0): Double? {
    if (sigmaHm == null || !sigmaHm.isFinite() || sigmaHm <= 0.0) return null
    return 1.0 - exp(-(radiusM * radiusM) / (2.0 * sigmaHm * sigmaHm))
}

/**
 * Android's Location accuracy is the radius of 68 % confidence. For a circular Gaussian
 * the 68 % radius is sigma * sqrt(-2 ln 0.32) = 1.5096 sigma, so sigma = accuracy / 1.5096.
 */
val ANDROID_ACCURACY_TO_SIGMA: Double = 1.0 / sqrt(-2.0 * ln(0.32))

fun sigmaFromAndroidAccuracy(accuracyM: Double): Double = accuracyM * ANDROID_ACCURACY_TO_SIGMA
