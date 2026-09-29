package com.sih26168.deadreckoning.core.nav

import com.sih26168.deadreckoning.core.Channel
import com.sih26168.deadreckoning.core.ChannelSample
import com.sih26168.deadreckoning.core.GnssSample
import com.sih26168.deadreckoning.core.GnssState
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Port of navcore.fusion.engine.NavigationEngine (causal mount; display = filter):
 *   WAITING    until a received fix shows speed > initMinSpeedMps with a bearing
 *   ALIGNING   8 navigators with vehicle-heading offsets k * 45 deg for mhWindowS; the
 *              lowest mean (capped) GNSS NIS + 50 per lock-out reset is kept
 *   NAVIGATING a single FusedNavigator
 * Tilt: accelerometer mean before start-up (>= 50 samples), else the phone's gravity channel.
 * The DISPLAYED position is the FILTER's (project decision after Phase 9).
 * Parity: nav_engine.json.
 */
data class EngineConfig(
    val fusion: FusionConfig = FusionConfig(),
    val initMinSpeedMps: Double = 5.0,
    val mhWindowS: Double = 120.0,
    val nHypotheses: Int = 8,
)

data class EngineOutput(
    val tS: Double,
    val mode: String,
    val latRad: Double,
    val lonRad: Double,
    val hM: Double,
    val sigmaHm: Double,
    val gnssState: GnssState,
    val vEnu: DoubleArray,
    val pDiag: DoubleArray,
)

object Mounting {
    /** Unit 'up' in the body frame from specific-force samples (>= 10, gravity-dominated). */
    fun upDirection(accel: List<DoubleArray>): DoubleArray {
        val a = accel.filter { it.all(Double::isFinite) }
        require(a.size >= 10) { "need >= 10 finite accelerometer samples to level" }
        val m = DoubleArray(3) { i -> a.sumOf { it[i] } / a.size }
        val n = norm(m)
        require(n >= 5.0) { "mean specific force $n m/s^2 is not gravity-dominated" }
        return scale(m, 1.0 / n)
    }

    /** Minimal rotation R_lb with R_lb up = [0, 0, 1] (Rodrigues). */
    fun tiltRotation(up: DoubleArray): Mat {
        val u = scale(up, 1.0 / norm(up))
        val z = doubleArrayOf(0.0, 0.0, 1.0)
        val axis = cross(u, z)
        val s = norm(axis)
        val c = dot(u, z)
        if (s < 1e-12) return if (c > 0) Mat.eye(3) else Mat.diag(doubleArrayOf(1.0, -1.0, -1.0))
        val k = scale(axis, 1.0 / s)
        val K = skew(k)
        val ang = atan2(s, c)
        return Mat.eye(3) + K * sin(ang) + (K * K) * (1 - cos(ang))
    }
}

class DrEngine(
    private val noise: NoiseParams,
    val cfg: EngineConfig = EngineConfig(),
    private val speedModel: SpeedModel? = null,
    private val axisMap: AxisMap = AxisMap.ANDROID_LIVE,
) {
    var mode = "WAITING"
        private set
    private val navs = mutableListOf<Pair<FusedNavigator, Double>>()
    private val accBuf = mutableListOf<DoubleArray>()
    private val gravBuf = mutableListOf<DoubleArray>()
    private var f0: GnssSample? = null
    var tInit: Double? = null
        private set
    var chosenOffsetRad: Double? = null
        private set
    var tiltSource: String? = null
        private set
    var mount: Mat? = null
        private set

    val nav: FusedNavigator? get() = if (navs.size == 1) navs[0].first else null

    private fun start(t: Double) {
        val f = f0!!
        val (Rvb, src) = if (accBuf.size >= 50) {
            Mounting.tiltRotation(Mounting.upDirection(accBuf)) to "accelerometer"
        } else {
            val data = gravBuf.ifEmpty { accBuf }
            val up = if (data.size >= 10) Mounting.upDirection(data) else {
                val m = DoubleArray(3) { i -> data.sumOf { it[i] } / data.size }
                scale(m, 1.0 / norm(m))
            }
            Mounting.tiltRotation(up) to (if (gravBuf.isEmpty()) "accelerometer" else "gravity_channel")
        }
        mount = Rvb
        tiltSource = src
        val offsets = List(cfg.nHypotheses) { it * 2 * PI / cfg.nHypotheses }
        val vehYaw = PI / 2 - f.bearingRad!!
        val sp = f.speedMps!!
        val v0 = doubleArrayOf(sp * sin(f.bearingRad), sp * cos(f.bearingRad), 0.0)
        val ys = Math.toRadians(25.0)  // the causal mount's yaw is never "acceptable"
        val acc = f.horizontalAccuracyM
        val p0 = doubleArrayOf(acc * acc, acc * acc, (2.5 * acc) * (2.5 * acc), 0.25, 0.25, 0.25,
            Math.toRadians(2.0) * Math.toRadians(2.0), Math.toRadians(2.0) * Math.toRadians(2.0), ys * ys,
            noise.accelBiasSigmaMps2 * noise.accelBiasSigmaMps2, noise.accelBiasSigmaMps2 * noise.accelBiasSigmaMps2,
            noise.accelBiasSigmaMps2 * noise.accelBiasSigmaMps2, noise.gyroBiasSigmaRadps * noise.gyroBiasSigmaRadps,
            noise.gyroBiasSigmaRadps * noise.gyroBiasSigmaRadps, noise.gyroBiasSigmaRadps * noise.gyroBiasSigmaRadps)
        for (off in offsets) {
            val nominal = NavState(t, f.latRad, f.lonRad, f.hM, v0, Quat.fromDcm(rotZ(vehYaw + off) * Rvb))
            val nav = FusedNavigator(ErrorStateEkf(nominal, Mat.diag(p0), noise), axisMap, cfg.fusion, speedModel,
                Rvb = Rvb, lateralNhcAllowed = false)
            navs += nav to off
        }
        tInit = t
        mode = "ALIGNING"
    }

    private fun score(nav: FusedNavigator): Double {
        val g = nav.ekf.log.filter { it.kind == "gnss" }.map { if (it.accepted) minOf(it.nis, 50.0) else 50.0 }
        return (if (g.isEmpty()) Double.POSITIVE_INFINITY else g.sum() / g.size) + 50.0 * nav.reset.nResets
    }

    /** One phone sample (10 Hz: see [TenHzResampler]) + the fixes received since the last one. */
    fun onSample(t: Double, acc: DoubleArray, grav: DoubleArray, gyro: DoubleArray, fixesIn: List<GnssSample> = emptyList()): EngineOutput? {
        var fixes = fixesIn
        if (mode == "WAITING") {
            accBuf += acc.copyOf()
            gravBuf += grav.copyOf()
            for (f in fixes) {
                if (f0 == null && f.speedMps != null && f.speedMps > cfg.initMinSpeedMps && f.bearingRad != null) f0 = f
            }
            val first = f0
            if (first == null || t < first.tReceivedS - 1e-9) return null
            start(t)
            fixes = fixes.filter { it.tReceivedS > t + 1e-9 }  // f0 initialises; it is not fused
        }
        for ((nav, _) in navs) nav.step(t, acc, grav, gyro, fixes)
        val ti = tInit!!
        if (mode == "ALIGNING" && navs.size > 1 && t >= ti + cfg.mhWindowS) {
            val best = navs.minByOrNull { score(it.first) }!!  // first of equal minima, like Python's min
            navs.clear()
            navs += best
        }
        if (navs.size == 1 && mode == "ALIGNING" && t >= ti + cfg.mhWindowS) {
            mode = "NAVIGATING"
            chosenOffsetRad = navs[0].second
        }
        val n = navs[0].first
        val s = n.ekf.nominal
        return EngineOutput(t, mode, s.latRad, s.lonRad, s.hM, sqrt(n.ekf.P[0, 0] + n.ekf.P[1, 1]), n.sm.state,
            s.vEnu.copyOf(), n.ekf.P.diag())
    }
}

/**
 * DEVICE-ONLY step (no Python counterpart): a live phone delivers sensors at ~50-200 Hz and
 * irregularly, but Model A was trained on, and the reference runs at, IO-VNBD's 10 Hz. This
 * averages ACCEL, GYRO and GRAVITY over consecutive [periodS] bins (a boxcar anti-alias) and
 * emits one sample per bin at the bin END. A bin missing a channel is SKIPPED and counted --
 * never filled with a guess. Whether IO-VNBD's 10 Hz logging averaged or decimated is not
 * documented; averaging is the conservative choice and is stated as a deviation to check
 * against on-device recordings.
 */
class TenHzResampler(private val periodS: Double = 0.1) {
    data class Sample(val tS: Double, val acc: DoubleArray, val grav: DoubleArray, val gyro: DoubleArray)

    private var binIndex: Long? = null  // bin (k-1, k] * period; ends from the INDEX, so they do not drift
    private val sums = HashMap<Channel, DoubleArray>()
    private val counts = HashMap<Channel, Int>()
    var nEmitted = 0L
        private set
    var nSkippedIncomplete = 0L
        private set

    /** Feed one channel sample; returns the finished 10 Hz sample(s), oldest first. */
    fun offer(s: ChannelSample): List<Sample> {
        val out = mutableListOf<Sample>()
        if (s.channel == Channel.MAG) return out
        var k = binIndex ?: (Math.floor(s.tS / periodS).toLong() + 1)
        while (s.tS >= k * periodS - 1e-12) {
            flush(k * periodS)?.let { out += it }
            k++
        }
        binIndex = k
        val sum = sums.getOrPut(s.channel) { DoubleArray(3) }
        sum[0] += s.value.x; sum[1] += s.value.y; sum[2] += s.value.z
        counts[s.channel] = (counts[s.channel] ?: 0) + 1
        return out
    }

    private fun flush(end: Double): Sample? {
        val have = listOf(Channel.ACCEL, Channel.GYRO, Channel.GRAVITY).all { (counts[it] ?: 0) > 0 }
        val r = if (have) {
            fun mean(c: Channel) = sums.getOrDefault(c, DoubleArray(3)).let { sum ->
                val cnt = counts[c] ?: 1
                DoubleArray(3) { i -> sum[i] / cnt }
            }
            nEmitted++
            Sample(end, acc = mean(Channel.ACCEL), grav = mean(Channel.GRAVITY), gyro = mean(Channel.GYRO))
        } else {
            if (counts.values.any { it > 0 }) nSkippedIncomplete++
            null
        }
        sums.clear()
        counts.clear()
        return r
    }
}
