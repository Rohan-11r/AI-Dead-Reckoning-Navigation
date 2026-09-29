package com.sih26168.deadreckoning.app

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

/*
 * Live, display-only monitors behind the map's "Real-Time AI Diagnostics" card. Both READ what
 * the engine does; neither feeds anything back into it.
 */

/**
 * One accelerometer spike: [rawPeakDevMps2] is the largest |‖a‖ - baseline| of any raw sample in
 * the 100 ms bin; [averagedDevMps2] is |‖bin mean‖ - baseline|, i.e. what is left of it in the
 * 10 Hz sample the INS integrates. [suppressed] = the averaged value is below the detection
 * threshold. [tS] = bin end, session seconds.
 */
data class ShockEvent(val tS: Double, val rawPeakDevMps2: Double, val averagedDevMps2: Double, val suppressed: Boolean)

/** Snapshot of [ShockMonitor] for the UI; [nowS] is the session time it was taken at. */
data class VibrationStatus(
    val active: Boolean = false, // the dead-reckoning engine (and so its 10 Hz averaging) is running
    val bins: Long = 0,
    val nShocks: Long = 0,
    val last: ShockEvent? = null,
    val nowS: Double = 0.0,
) {
    /** True for [holdS] after the latest spike: the UI flashes its warning. */
    fun flashing(holdS: Double = SHOCK_FLASH_S): Boolean = last?.let { nowS - it.tS in 0.0..holdS } ?: false
}

const val SHOCK_FLASH_S = 1.5

/**
 * Detects short accelerometer spikes (potholes, kerbs, knocks on the mount) in the RAW device
 * stream and measures how much of each survives the engine's 10 Hz boxcar averaging
 * ([com.sih26168.deadreckoning.core.nav.TenHzResampler], same bin boundaries: (k-1, k] x
 * [periodS] on the session clock), so the bin mean here is the accelerometer value the INS gets.
 *
 * Baseline: an exponential average (time constant [baselineTauS]) of the bin-mean specific-force
 * magnitude -- gravity plus sustained manoeuvres, which are navigation, not shocks. A bin is a
 * shock when a raw sample deviates from it by more than [thresholdMps2]; detection starts after
 * [warmupBins] bins. The threshold is a starting value, NOT tuned on recorded drives yet.
 */
class ShockMonitor(
    private val periodS: Double = 0.1,
    private val thresholdMps2: Double = 4.0,
    private val baselineTauS: Double = 1.0,
    private val warmupBins: Int = 10,
) {
    private var binIndex: Long? = null
    private val sum = DoubleArray(3)
    private var n = 0
    private var maxNorm = Double.NEGATIVE_INFINITY
    private var minNorm = Double.POSITIVE_INFINITY
    private var baseline: Double? = null
    var nBins = 0L
        private set
    var nShocks = 0L
        private set
    var last: ShockEvent? = null
        private set

    /** One raw accelerometer sample, device frame, m/s^2, at session time [tS]. */
    fun offer(tS: Double, xMps2: Double, yMps2: Double, zMps2: Double) {
        var k = binIndex ?: (floor(tS / periodS).toLong() + 1)
        while (tS >= k * periodS - 1e-12) { // the resampler's rule: close every bin this sample is past
            flush(k * periodS)
            k++
        }
        binIndex = k
        sum[0] += xMps2; sum[1] += yMps2; sum[2] += zMps2
        n++
        val norm = sqrt(xMps2 * xMps2 + yMps2 * yMps2 + zMps2 * zMps2)
        if (norm > maxNorm) maxNorm = norm
        if (norm < minNorm) minNorm = norm
    }

    private fun flush(endS: Double) {
        if (n == 0) return
        val mx = sum[0] / n
        val my = sum[1] / n
        val mz = sum[2] / n
        val meanNorm = sqrt(mx * mx + my * my + mz * mz)
        val b = baseline
        if (b != null && nBins >= warmupBins) {
            val peak = maxOf(abs(maxNorm - b), abs(minNorm - b))
            if (peak > thresholdMps2) {
                val averaged = abs(meanNorm - b)
                nShocks++
                last = ShockEvent(endS, peak, averaged, suppressed = averaged < thresholdMps2)
            }
        }
        baseline = if (b == null) meanNorm else b + (periodS / baselineTauS) * (meanNorm - b)
        nBins++
        sum.fill(0.0)
        n = 0
        maxNorm = Double.NEGATIVE_INFINITY
        minNorm = Double.POSITIVE_INFINITY
    }

    fun status(nowS: Double, active: Boolean) = VibrationStatus(active, nBins, nShocks, last, nowS)
}

/** What the engine's non-holonomic constraint (NHC) update is doing, from its own counters. */
data class NhcStatus(val phase: Phase = Phase.UNAVAILABLE, val acceptedTotal: Long = 0) {
    enum class Phase {
        UNAVAILABLE,    // no dead-reckoning engine
        OFF_NO_MODEL,   // engine without the speed model: NHC is off by configuration
        WAITING,        // engine initialising / aligning its heading: no single navigator yet
        APPLIED,        // last NHC update accepted by the filter
        GATED,          // last NHC update rejected by the innovation gate
        LOW_SPEED,      // skipped: below the NHC minimum speed
        HARD_CORNERING, // skipped: lateral acceleration too high for the constraint
    }
}

/**
 * Turns the navigator's cumulative `nhc_*` counters (FusedNavigator.counts) into the latest
 * outcome: the counter that grew since the previous call. With no growth the previous outcome
 * is kept (a publish can fall between two 10 Hz steps). A counter that shrank (a new navigator
 * after alignment) counts from zero.
 */
class NhcMonitor {
    private var prev: Map<String, Long> = emptyMap()
    private var phase = NhcStatus.Phase.WAITING

    fun update(counts: Map<String, Long>?): NhcStatus {
        if (counts == null) {
            prev = emptyMap()
            phase = NhcStatus.Phase.WAITING
            return NhcStatus(phase, 0)
        }
        var best: NhcStatus.Phase? = null
        var bestDelta = 0L
        for ((key, p) in KEYS) {
            val cur = counts[key] ?: 0L
            val before = prev[key] ?: 0L
            val delta = if (cur >= before) cur - before else cur
            if (delta > bestDelta) {
                bestDelta = delta
                best = p
            }
        }
        if (best != null) phase = best
        prev = KEYS.keys.associateWith { counts[it] ?: 0L }
        return NhcStatus(phase, counts["nhc_accepted"] ?: 0L)
    }

    private companion object {
        val KEYS = linkedMapOf(
            "nhc_accepted" to NhcStatus.Phase.APPLIED,
            "nhc_gated" to NhcStatus.Phase.GATED,
            "nhc_low_speed" to NhcStatus.Phase.LOW_SPEED,
            "nhc_hard_cornering" to NhcStatus.Phase.HARD_CORNERING,
        )
    }
}
