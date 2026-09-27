package com.sih26168.deadreckoning.core.nav

import com.sih26168.deadreckoning.core.GnssSample
import com.sih26168.deadreckoning.core.GnssState
import com.sih26168.deadreckoning.core.GnssStateMachine
import com.sih26168.deadreckoning.core.POLICY
import com.sih26168.deadreckoning.core.FilterPolicy
import kotlin.math.abs
import kotlin.math.hypot

/**
 * Port of navcore.fusion.navigator.FusedNavigator with the SHIPPED defaults
 * (FusionConfig(): AI speed ON, NHC ON only with AI speed, Model B OFF, state machine ON,
 * recovery manager OFF). One step per phone sample, the reference's order:
 *   axis map -> window (+ model when ready) -> predict (sample at the START of the
 *   interval) -> state machine + GNSS (+ lock-out reset) -> AI speed (fresh, <= 1 Hz,
 *   variance x window/interval) -> NHC -> stillness ZUPT + levelling.
 * Model B and the recovery manager are not ported (OFF by default, Phases 7 and 9).
 * Parity: nav_navigator.json.
 */
data class FusionConfig(
    val useAiSpeed: Boolean = true,
    val useNhc: Boolean = true,
    val nhcRequiresAiSpeed: Boolean = true,
    val useStateMachine: Boolean = true,
    val aiEvery: Int = 1,
    val aiSpeedMinMps: Double = 1.0,
    val aiSpeedSigmaFloorMps: Double = 0.5,
    val aiSpeedIntervalS: Double = 1.0,
    val aiSampleDtS: Double = 0.1,
    val stillAccStdMps2: Double = 0.08,
    val stillNormTolMps2: Double = 0.1,
    val stillGyroRadps: Double = 0.02,
    val stillSpeedMps: Double = 1.0,
    val nhc: NhcConfig = NhcConfig(),
)

/** Raw logged columns -> BODY: body = M raw (signed permutation), per-axis gyro validity. */
class AxisMap(val accel: Mat, val gyro: Mat, val gyroValid: BooleanArray) {
    fun toBody(accRaw: DoubleArray, gyroRaw: DoubleArray): Pair<DoubleArray, DoubleArray> =
        accel.mulVec(accRaw) to gyro.mulVec(gyroRaw)

    companion object {
        /** IO-VNBD phone logs: gyro column 2 is the vertical rate; columns 1/3 are not rates. */
        val IOVNBD = AxisMap(Mat.eye(3), Mat.of(3, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0), booleanArrayOf(false, false, true))

        /** A live Android phone: all three gyro axes are real rates (DEVICE frame = body). */
        val ANDROID_LIVE = AxisMap(Mat.eye(3), Mat.eye(3), booleanArrayOf(true, true, true))
    }
}

class FusedNavigator(
    val ekf: ErrorStateEkf,
    private val axisMap: AxisMap,
    val cfg: FusionConfig = FusionConfig(),
    private val modelSpeed: SpeedModel? = null,
    Rvb: Mat? = null,
    lateralNhcAllowed: Boolean = false,
    private val gamma: Double = 9.81,
) {
    private val rvb: Mat = Rvb ?: Mat.eye(3)
    private val lateralOk = lateralNhcAllowed && Rvb != null
    val sm = GnssStateMachine()
    val reset = ConsecutiveRejectionReset()
    val counts = linkedMapOf<String, Long>()
    private val win = ImuWindow(modelSpeed?.window ?: 1)
    private val accHist = ArrayDeque<Double>()
    private var prev: Triple<Double, DoubleArray, DoubleArray>? = null  // (t, acc raw, gyro raw)
    private var n = 0L
    var aiSpeed: Pair<Double, Double>? = null
        private set
    private var aiSpeedFresh = false
    private var aiSpeedLastT = Double.NEGATIVE_INFINITY
    private val aiSpeedInflation = maxOf(1.0, (modelSpeed?.window ?: 1) * cfg.aiSampleDtS / cfg.aiSpeedIntervalS)
    var lastFixSpeed: Double? = null
        private set

    private fun count(k: String) {
        counts[k] = (counts[k] ?: 0L) + 1L
    }

    private fun policy(): FilterPolicy =
        if (!cfg.useStateMachine) FilterPolicy(useGnss = true, gnssNoiseInflation = 1.0, useAiSpeed = false, useNhc = true) else sm.policy

    private fun still(accNorm: Double, wz: Double): Boolean {
        accHist.addLast(accNorm)
        if (accHist.size > 10) accHist.removeFirst()
        if (accHist.size < 10) return false
        val hint = aiSpeed?.first ?: lastFixSpeed
        return populationStd(accHist.toList()) < cfg.stillAccStdMps2 && abs(accNorm - gamma) < cfg.stillNormTolMps2 &&
            abs(wz) < cfg.stillGyroRadps && hint != null && hint < cfg.stillSpeedMps
    }

    /** One phone sample [acc] (m/s^2), [grav] (m/s^2), [gyroRaw] (rad/s, logged columns) +
     * the fixes RECEIVED since the previous sample. */
    fun step(t: Double, acc: DoubleArray, grav: DoubleArray, gyroRaw: DoubleArray, fixes: List<GnssSample> = emptyList()) {
        val (accB, gyroB) = axisMap.toBody(acc, gyroRaw)
        val wz = gyroB[2]
        val p = prev
        prev = Triple(t, acc.copyOf(), gyroRaw.copyOf())
        win.push(t, accB, grav, wz)
        n++
        val model = modelSpeed
        if (model != null && win.ready() && n % cfg.aiEvery == 0L) {
            val (a, g, w) = win.last(model.window)
            aiSpeed = model.predict(a, g, w)
            count("ai_speed_inferences")
            aiSpeedFresh = true
        }
        if (p == null) return
        val dt = t - p.first
        if (dt <= 0) {
            count("skipped_nonpositive_dt")
            return
        }
        val (pa, pg) = axisMap.toBody(p.second, p.third)
        ekf.predict(FilterImu(p.first, pa, pg, axisMap.gyroValid), dt)
        updates(t, fixes, accB, wz, norm(acc))
    }

    private fun fuseGnss(t: Double, fix: GnssSample) {
        var pol = policy()
        if (cfg.useStateMachine && sm.state == GnssState.LOST) pol = POLICY.getValue(GnssState.RECOVERING)
        if (!pol.useGnss) return
        val f = fix.copy(horizontalAccuracyM = fix.horizontalAccuracyM * pol.gnssNoiseInflation)
        val ok = ekf.updateGnss(f)
        count(if (ok) "gnss_accepted" else "gnss_rejected")
        if (reset.afterUpdate(ekf, fix, ok)) count("gnss_lockout_resets")
        if (cfg.useStateMachine) sm.onFix(t, fix.horizontalAccuracyM, ok)
    }

    private fun updates(t: Double, fixes: List<GnssSample>, accB: DoubleArray, wz: Double, accNorm: Double) {
        if (cfg.useStateMachine) sm.onTime(t)
        for (fix in fixes) {
            if (fix.speedMps != null) lastFixSpeed = fix.speedMps
            fuseGnss(t, fix)
        }
        val pol = policy()

        val ai = aiSpeed
        if (cfg.useAiSpeed && pol.useAiSpeed && ai != null && aiSpeedFresh && t - aiSpeedLastT >= cfg.aiSpeedIntervalS - 1e-9) {
            aiSpeedFresh = false
            aiSpeedLastT = t
            val inn = speedInnovation(ekf, ai.first, cfg.aiSpeedMinMps)
            if (inn == null) {
                count("ai_speed_skipped_low_speed")
            } else {
                val sig2 = maxOf(ai.second, cfg.aiSpeedSigmaFloorMps * cfg.aiSpeedSigmaFloorMps) * aiSpeedInflation
                val ok = ekf.update("ai_speed", t, inn.first, inn.second, Mat.of(1, 1, sig2))
                count(if (ok) "ai_speed_accepted" else "ai_speed_rejected")
            }
        }

        if (cfg.useNhc && pol.useNhc && (cfg.useAiSpeed || !cfg.nhcRequiresAiSpeed)) {
            val v = ekf.nominal.vEnu
            val latAcc = hypot(v[0], v[1]) * wz
            count("nhc_" + Nhc.apply(ekf, t, rvb, lateralOk, latAcc, cfg.nhc))
        }

        if (still(accNorm, wz)) {
            ekf.updateZupt(t)
            ekf.updateLevelling(t, accB)
            count("zupt")
        }
    }
}
