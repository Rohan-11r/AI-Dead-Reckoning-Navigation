package com.sih26168.deadreckoning.core.nav

import com.sih26168.deadreckoning.core.GnssSample
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Port of navcore.filtering.ekf: the 15-state error-state EKF.
 * dx = [dp_ENU(3), dv_ENU(3), psi(3), db_a(3), db_g(3)]; errors are TRUTH MINUS ESTIMATE;
 * R_b^n = (I + [psi]x) R_hat. Joseph-form updates, symmetrisation and a Cholesky check
 * after every step (a broken P throws [CovarianceError]). GNSS innovations are formed
 * against the state at the fix EPOCH (short history). Parity: nav_ekf_ops.json.
 */
const val N_STATES = 15
const val P0I = 0
const val V0I = 3
const val A0I = 6
const val BA0I = 9
const val BG0I = 12

/** chi-square 99.9 % quantiles for NIS gating, by dimension (the reference's table). */
val CHI2_999 = mapOf(1 to 10.828, 2 to 13.816, 3 to 16.266, 4 to 18.467, 5 to 20.515, 6 to 22.458)

class CovarianceError(message: String) : RuntimeException(message)

data class NoiseParams(
    val accelWhiteMps2RtHz: Double,
    val gyroWhiteRadpsRtHz: Double,
    val accelBiasSigmaMps2: Double,
    val accelBiasTauS: Double,
    val gyroBiasSigmaRadps: Double,
    val gyroBiasTauS: Double,
    val unmeasuredRateRadpsRtHz: Double = 0.02,
)

data class UpdateRecord(val kind: String, val tS: Double, val accepted: Boolean, val nis: Double, val dim: Int)

/** An IMU sample for the filter: specific force and rate in BODY, with per-axis gyro validity. */
class FilterImu(val tS: Double, val accel: DoubleArray, val gyro: DoubleArray, val gyroValid: BooleanArray) {
    init {
        for (i in 0..2) require(gyroValid[i] || gyro[i] == 0.0) { "invalid gyro axes must be zero" }
    }
}

class ErrorStateEkf(
    nominal: NavState,
    p0: Mat,
    val noise: NoiseParams,
    val historyS: Double = 15.0,
) {
    var nominal: NavState = nominal
        internal set
    var P: Mat = p0.copy()
        internal set
    var ba = DoubleArray(3)
        internal set
    var bg = DoubleArray(3)
        internal set
    val log = mutableListOf<UpdateRecord>()
    internal val hist = ArrayDeque<NavState>()

    init {
        require(P.rows == N_STATES && P.cols == N_STATES)
        checkP("init")
        hist.addLast(this.nominal)
    }

    internal fun checkP(where: String) {
        if (!P.isFinite()) throw CovarianceError("P non-finite after $where")
        P = (P + P.t()) * 0.5
        try {
            cholesky(P)
        } catch (x: NotPositiveDefinite) {
            throw CovarianceError("P not positive-definite after $where")
        }
    }

    fun jacobianF(fB: DoubleArray, gyroValid: BooleanArray): Mat {
        val s = nominal
        val R = Quat.toDcm(s.qNb)
        val fN = R.mulVec(fB)
        val wIe = Geodesy.earthRateEnu(s.latRad)
        val wEn = Geodesy.transportRateEnu(s.latRad, s.hM, s.vEnu)
        val F = Mat(N_STATES, N_STATES)
        F.setBlock(P0I, V0I, Mat.eye(3))
        F.setBlock(V0I, V0I, skew(add(scale(wIe, 2.0), wEn)) * -1.0)
        F.setBlock(V0I, A0I, skew(fN) * -1.0)
        F.setBlock(V0I, BA0I, R * -1.0)
        F[5, 2] = Const.FREE_AIR_GRADIENT_PER_S2
        F.setBlock(A0I, A0I, skew(add(wIe, wEn)) * -1.0)
        val rg = R * -1.0
        for (i in 0..2) for (j in 0..2) if (!gyroValid[j]) rg[i, j] = 0.0 * rg[i, j]
        F.setBlock(A0I, BG0I, rg)
        for (i in 0..2) {
            F[BA0I + i, BA0I + i] = -1.0 / noise.accelBiasTauS
            F[BG0I + i, BG0I + i] = -1.0 / noise.gyroBiasTauS
        }
        return F
    }

    private fun q(dt: Double, gyroValid: BooleanArray): Mat {
        val n = noise
        val R = Quat.toDcm(nominal.qNb)
        val gPsd = DoubleArray(3) { val s = if (gyroValid[it]) n.gyroWhiteRadpsRtHz else n.unmeasuredRateRadpsRtHz; s * s }
        val Q = Mat(N_STATES, N_STATES)
        val qa = n.accelWhiteMps2RtHz * n.accelWhiteMps2RtHz * dt
        for (i in 0..2) Q[V0I + i, V0I + i] = qa
        Q.setBlock(A0I, A0I, R * Mat.diag(gPsd) * R.t() * dt)
        val qba = 2.0 * n.accelBiasSigmaMps2 * n.accelBiasSigmaMps2 / n.accelBiasTauS * dt
        val qbg = 2.0 * n.gyroBiasSigmaRadps * n.gyroBiasSigmaRadps / n.gyroBiasTauS * dt
        for (i in 0..2) {
            Q[BA0I + i, BA0I + i] = qba
            Q[BG0I + i, BG0I + i] = if (gyroValid[i]) qbg else 0.0
        }
        return Q
    }

    /** Propagate nominal state and covariance over [dt] with [imu] (the sample at the START). */
    fun predict(imu: FilterImu, dt: Double) {
        val valid = imu.gyroValid
        val fB = sub(imu.accel, ba)
        val s = nominal
        val wInB = Quat.toDcm(s.qNb).t().mulVec(add(Geodesy.earthRateEnu(s.latRad), Geodesy.transportRateEnu(s.latRad, s.hM, s.vEnu)))
        val wB = DoubleArray(3) { if (valid[it]) imu.gyro[it] - bg[it] else wInB[it] }
        val F = jacobianF(fB, valid)
        val Fd = F * dt
        val Phi = Mat.eye(N_STATES) + Fd + (Fd * Fd) * 0.5
        P = Phi * P * Phi.t() + q(dt, valid)
        for (i in 0..2) if (!valid[i]) {
            val k = BG0I + i
            for (j in 0 until N_STATES) { P[k, j] = 0.0; P[j, k] = 0.0 }
            P[k, k] = noise.gyroBiasSigmaRadps * noise.gyroBiasSigmaRadps
        }
        nominal = Ins.propagate(nominal, fB, wB, dt)
        hist.addLast(nominal)
        while (hist.isNotEmpty() && hist.first().tS < nominal.tS - historyS) hist.removeFirst()
        checkP("predict")
    }

    /** Generic update; nu = z - h(x_hat) = H dx. NIS-gated, Joseph form, PD-checked, logged. */
    fun update(kind: String, tS: Double, nu: DoubleArray, H: Mat, Rm: Mat, gate: Boolean = true): Boolean {
        require(H.rows == nu.size && H.cols == N_STATES && Rm.rows == nu.size && Rm.cols == nu.size) {
            "update $kind: inconsistent shapes"
        }
        val S = H * P * H.t() + Rm
        val nis = dot(nu, solveVec(S, nu))
        val dim = nu.size
        if (gate && nis > CHI2_999.getValue(dim)) {
            log += UpdateRecord(kind, tS, false, nis, dim)
            return false
        }
        val K = solve(S, H * P).t()  // P H^T S^-1 (S symmetric)
        val dx = K.mulVec(nu)
        val IKH = Mat.eye(N_STATES) - K * H
        P = IKH * P * IKH.t() + K * Rm * K.t()
        inject(dx)
        checkP("update:$kind")
        log += UpdateRecord(kind, tS, true, nis, dim)
        return true
    }

    /** Absorb dx into the nominal state; R <- (I + [psi]x) R_hat. */
    internal fun inject(dx: DoubleArray) {
        val s = nominal
        val rm = Geodesy.meridianRadius(s.latRad) + s.hM
        val rn = Geodesy.primeVerticalRadius(s.latRad) + s.hM
        val q = Quat.normalize(Quat.mul(Quat.fromRotvec(doubleArrayOf(dx[A0I], dx[A0I + 1], dx[A0I + 2])), s.qNb))
        nominal = NavState(s.tS, s.latRad + dx[1] / rm, s.lonRad + dx[0] / (rn * cos(s.latRad)), s.hM + dx[2],
            DoubleArray(3) { s.vEnu[it] + dx[V0I + it] }, q)
        ba = DoubleArray(3) { ba[it] + dx[BA0I + it] }
        bg = DoubleArray(3) { bg[it] + dx[BG0I + it] }
        if (hist.isNotEmpty()) {
            hist.removeLast()
            hist.addLast(nominal)
        }
    }

    /** The history state closest in time to [tS] (first of equals, like Python's min). */
    internal fun stateAt(tS: Double): NavState {
        if (hist.isEmpty()) return nominal
        var best = hist.first()
        var d = kotlin.math.abs(best.tS - tS)
        for (s in hist) {
            val e = kotlin.math.abs(s.tS - tS)
            if (e < d) { d = e; best = s }
        }
        return best
    }

    class Innovation(val nu: DoubleArray, val H: Mat, val R: Mat)

    fun gnssInnovation(
        fix: GnssSample, verticalSigmaFactor: Double = 2.5, velocitySigmaMps: Double? = 0.5, minSpeedForVelocityMps: Double = 2.0,
    ): Innovation {
        require(fix.tReceivedS <= nominal.tS + 1e-9) { "fix used before it was received (non-causal)" }
        val ref = stateAt(fix.tS)
        val rm = Geodesy.meridianRadius(ref.latRad) + ref.hM
        val rn = Geodesy.primeVerticalRadius(ref.latRad) + ref.hM
        val dlon = Geodesy.wrapPi(fix.lonRad - ref.lonRad)
        val nu = mutableListOf(dlon * rn * cos(ref.latRad), (fix.latRad - ref.latRad) * rm, fix.hM - ref.hM)
        val sig = fix.horizontalAccuracyM
        val variances = mutableListOf(sig * sig, sig * sig, (verticalSigmaFactor * sig) * (verticalSigmaFactor * sig))
        val rows = mutableListOf<DoubleArray>()
        for (i in 0..2) rows += DoubleArray(N_STATES).also { it[i] = 1.0 }
        val sp = fix.speedMps
        val br = fix.bearingRad
        if (velocitySigmaMps != null && sp != null && br != null && sp >= minSpeedForVelocityMps) {
            nu += sp * sin(br) - ref.vEnu[0]
            nu += sp * cos(br) - ref.vEnu[1]
            variances += velocitySigmaMps * velocitySigmaMps
            variances += velocitySigmaMps * velocitySigmaMps
            rows += DoubleArray(N_STATES).also { it[3] = 1.0 }
            rows += DoubleArray(N_STATES).also { it[4] = 1.0 }
        }
        val H = Mat(rows.size, N_STATES, rows.flatMap { it.toList() }.toDoubleArray())
        return Innovation(nu.toDoubleArray(), H, Mat.diag(variances.toDoubleArray()))
    }

    fun updateGnss(fix: GnssSample): Boolean {
        val i = gnssInnovation(fix)
        return update("gnss", fix.tS, i.nu, i.H, i.R)
    }

    fun updateZupt(tS: Double, sigmaMps: Double = 0.05): Boolean {
        val H = Mat(3, N_STATES)
        for (i in 0..2) H[i, V0I + i] = 1.0
        return update("zupt", tS, scale(nominal.vEnu, -1.0), H, Mat.eye(3) * (sigmaMps * sigmaMps))
    }

    fun levellingInnovation(accel: DoubleArray): Innovation {
        val s = nominal
        val R = Quat.toDcm(s.qNb)
        val fN = R.mulVec(sub(accel, ba))
        val g = Geodesy.gravityEnu(s.latRad, s.hM)
        val nu = DoubleArray(3) { -(fN[it] + g[it]) }
        val H = Mat(3, N_STATES)
        H.setBlock(0, A0I, skew(fN) * -1.0)
        H.setBlock(0, BA0I, R * -1.0)
        return Innovation(nu, H, Mat.eye(3))
    }

    fun updateLevelling(tS: Double, accel: DoubleArray, sigmaMps2: Double = 0.2): Boolean {
        val i = levellingInnovation(accel)
        return update("level", tS, i.nu, i.H, Mat.eye(3) * (sigmaMps2 * sigmaMps2))
    }

    fun std(): DoubleArray = DoubleArray(N_STATES) { sqrt(P[it, it]) }
}

/** Port of navcore.recovery.gnss_reset.reset_to_fix. */
fun resetToFix(
    ekf: ErrorStateEkf, fix: GnssSample, verticalSigmaFactor: Double = 2.5, velocitySigmaMps: Double = 1.0,
    attitudeInflation: Double = 4.0, maxAttitudeSigmaRad: Double = 0.5,
) {
    val s = ekf.nominal
    val ref = ekf.stateAt(fix.tS)
    val lat = fix.latRad + (s.latRad - ref.latRad)
    val lon = fix.lonRad + (s.lonRad - ref.lonRad)
    val h = fix.hM + (s.hM - ref.hM)
    val v = s.vEnu.copyOf()
    val sp = fix.speedMps
    val br = fix.bearingRad
    val hasV = sp != null && br != null
    if (hasV) {
        v[0] = sp!! * sin(br!!)
        v[1] = sp * cos(br)
    }
    ekf.nominal = NavState(s.tS, lat, lon, h, v, s.qNb)
    val P = ekf.P.copy()
    for (i in 0..5) for (j in 0 until N_STATES) { P[i, j] = 0.0; P[j, i] = 0.0 }
    val sig = fix.horizontalAccuracyM
    P[0, 0] = sig * sig
    P[1, 1] = sig * sig
    P[2, 2] = (verticalSigmaFactor * sig) * (verticalSigmaFactor * sig)
    val vv = if (hasV) velocitySigmaMps * velocitySigmaMps else 25.0
    P[3, 3] = vv
    P[4, 4] = vv
    P[5, 5] = 1.0
    val std = DoubleArray(3) { sqrt(P[A0I + it, A0I + it]) }
    val newStd = DoubleArray(3) { minOf(std[it] * sqrt(attitudeInflation), maxAttitudeSigmaRad) }
    for (i in 0..2) for (j in 0..2) {
        P[A0I + i, A0I + j] = P[A0I + i, A0I + j] * ((newStd[i] / std[i]) * (newStd[j] / std[j]))
    }
    ekf.P = P
    ekf.checkP("reset_to_fix")
    ekf.hist.clear()
    ekf.hist.addLast(ekf.nominal)
    ekf.log += UpdateRecord("reset", fix.tS, true, Double.NaN, 0)
}

/** Port of navcore.recovery.gnss_reset.ConsecutiveRejectionReset. */
class ConsecutiveRejectionReset(private val nConsecutive: Int = 3) {
    var streak = 0
        private set
    var nResets = 0
        private set

    fun afterUpdate(ekf: ErrorStateEkf, fix: GnssSample, accepted: Boolean): Boolean {
        if (accepted) {
            streak = 0
            return false
        }
        streak++
        if (streak < nConsecutive) return false
        resetToFix(ekf, fix)
        streak = 0
        nResets++
        return true
    }
}
