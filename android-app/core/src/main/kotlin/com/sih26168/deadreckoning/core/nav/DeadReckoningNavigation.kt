package com.sih26168.deadreckoning.core.nav

import com.sih26168.deadreckoning.core.ChannelSample
import com.sih26168.deadreckoning.core.GnssSample
import com.sih26168.deadreckoning.core.GnssState
import com.sih26168.deadreckoning.core.ImuSample
import com.sih26168.deadreckoning.core.NavSnapshot
import com.sih26168.deadreckoning.core.NavigationEngine
import com.sih26168.deadreckoning.core.mapmatch.DeadReckoningMapMatcher
import com.sih26168.deadreckoning.core.mapmatch.MapMatchResult
import com.sih26168.deadreckoning.core.mapmatch.RoadNetwork
import com.sih26168.deadreckoning.core.positionConfidence
import kotlin.math.hypot

/**
 * The on-device DEAD-RECKONING engine behind the app's [NavigationEngine] contract (Phase 13 Part B):
 *   raw ChannelSamples -> [TenHzResampler] -> [DrEngine] (INS + EKF + Model A speed + NHC +
 *   GNSS state machine, heading hypotheses) -> optional [DeadReckoningMapMatcher].
 * GNSS fixes are queued and handed to the engine with the first 10 Hz sample at or after
 * their RECEIPT time (causality, as in the replay engine). The displayed position is the
 * FILTER's (project decision after Phase 9); the map-matched position is reported alongside
 * it, never instead of it, and never fed back (refinement only).
 * [onOutput] sees every 10 Hz engine output (used by the parity test and the app's track).
 */
class DeadReckoningNavigation(
    noise: NoiseParams,
    cfg: EngineConfig = EngineConfig(),
    private val speedModel: OnnxSpeedModel? = null,
    road: RoadNetwork? = null,
    axisMap: AxisMap = AxisMap.ANDROID_LIVE,
    testModel: SpeedModel? = null,
) : NavigationEngine {
    private val hasModel = speedModel != null || testModel != null
    override val name: String
        get() = buildString {
            append("DR engine (INS + EKF")
            if (hasModel) append(" + Model A speed + NHC") else append("; no speed model: AI speed and NHC off")
            append(if (matcher != null) " + map matching)" else "; no offline map)")
        }
    val engine = DrEngine(noise, cfg, speedModel ?: testModel, axisMap)
    val resampler = TenHzResampler()
    private var matcher: DeadReckoningMapMatcher? = road?.let { DeadReckoningMapMatcher(it) }
    val hasMap: Boolean get() = matcher != null

    /** Attach the offline road network later (it loads in the background on a phone).
     * Call from the same thread/coroutine that feeds the engine. */
    fun attachRoads(net: RoadNetwork) {
        matcher = DeadReckoningMapMatcher(net)
    }
    private val pending = ArrayDeque<GnssSample>()
    var last: EngineOutput? = null
        private set
    var lastMatch: MapMatchResult? = null
        private set
    var onOutput: ((EngineOutput) -> Unit)? = null

    override fun onImu(sample: ImuSample) {}  // the engine consumes resampled channels, not assembled IMU samples

    override fun onChannel(sample: ChannelSample) {
        for (s in resampler.offer(sample)) step(s)
    }

    override fun onFix(fix: GnssSample) {
        pending.addLast(fix)
    }

    private fun step(s: TenHzResampler.Sample) {
        val ready = ArrayList<GnssSample>()
        while (pending.isNotEmpty() && pending.first().tReceivedS <= s.tS + 1e-9) ready += pending.removeFirst()
        val out = engine.onSample(s.tS, s.acc, s.grav, s.gyro, ready) ?: return
        last = out
        onOutput?.invoke(out)
        val m = matcher
        val nav = engine.nav
        if (m != null && out.mode == "NAVIGATING" && nav != null) {
            val P = nav.ekf.P
            m.update(out.tS, out.latRad, out.lonRad, doubleArrayOf(P[0, 0], P[0, 1], P[1, 0], P[1, 1]),
                doubleArrayOf(out.vEnu[0], out.vEnu[1]), doubleArrayOf(P[3, 3], P[3, 4], P[4, 3], P[4, 4]))
                ?.let { lastMatch = it }
        }
    }

    override fun snapshot(): NavSnapshot {
        val o = last
        if (o == null) {
            return NavSnapshot(0.0, GnssState.LOST, null, null, null, null, null, null, null, speedModel?.lastLatencyMs,
                name, note = "waiting for a moving GNSS fix (> 5 m/s) to initialise", mode = engine.mode)
        }
        val mm = lastMatch?.takeIf { it.matched && it.tS >= o.tS - 2.0 }  // stale matches are not shown
        val note = when {
            o.mode == "ALIGNING" -> "aligning heading (8 hypotheses, ${engine.cfg.mhWindowS.toInt()} s): position provisional"
            o.gnssState == GnssState.LOST -> "GNSS lost: dead reckoning (INS + AI speed + NHC)"
            else -> null
        }
        return NavSnapshot(
            tS = o.tS, state = o.gnssState, latRad = o.latRad, lonRad = o.lonRad,
            speedMps = hypot(o.vEnu[0], o.vEnu[1]), sigmaHm = o.sigmaHm, confidence = positionConfidence(o.sigmaHm),
            roadName = mm?.roadName, covarianceDiag = o.pDiag, aiLatencyMs = speedModel?.lastLatencyMs?.takeIf { it.isFinite() },
            engineName = name, note = note, matchedLatRad = mm?.latRad, matchedLonRad = mm?.lonRad,
            mapMatchConfidence = mm?.confidence, mode = o.mode,
        )
    }
}
