package com.sih26168.deadreckoning.core.nav

import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Ports of navcore.nhc.constraints (NHC), navcore.fusion.features (model input), and the
 * windowing / speed-measurement helpers of navcore.fusion.ai_models.
 * Parity: nav_ekf_ops.json (NHC), nav_features.json, nav_model_a.json, nav_navigator.json.
 */
data class NhcConfig(
    val sigmaLateralMps: Double = 0.3,
    val sigmaVerticalMps: Double = 0.3,
    val minSpeedMps: Double = 2.0,
    val maxLateralAccelMps2: Double = 3.0,
)

object Nhc {
    /** (nu, H) for vehicle-frame rows (1 = lateral, 2 = vertical). */
    fun innovation(ekf: ErrorStateEkf, Rvb: Mat, rows: IntArray): Pair<DoubleArray, Mat> {
        val s = ekf.nominal
        val Rnb = Quat.toDcm(s.qNb)
        val v = s.vEnu
        val vB = Rnb.t().mulVec(v)
        val S = Mat(rows.size, 3)
        for ((k, r) in rows.withIndex()) for (j in 0..2) S[k, j] = Rvb[r, j]
        val nu = DoubleArray(rows.size) { k -> -dot(DoubleArray(3) { S[k, it] }, vB) }
        val H = Mat(rows.size, N_STATES)
        H.setBlock(0, V0I, S * Rnb.t())
        H.setBlock(0, A0I, S * Rnb.t() * skew(v))
        return nu to H
    }

    /** Vertical (+ lateral when allowed) NHC. Returns the reason, for counting. */
    fun apply(ekf: ErrorStateEkf, tS: Double, Rvb: Mat, lateralAllowed: Boolean, lateralAccelMps2: Double, cfg: NhcConfig): String {
        val v = ekf.nominal.vEnu
        val speed = hypot(v[0], v[1])
        if (speed < cfg.minSpeedMps) return "low_speed"
        if (kotlin.math.abs(lateralAccelMps2) > cfg.maxLateralAccelMps2) return "hard_cornering"
        val rows = if (lateralAllowed) intArrayOf(1, 2) else intArrayOf(2)
        val sig = if (lateralAllowed) doubleArrayOf(cfg.sigmaLateralMps, cfg.sigmaVerticalMps) else doubleArrayOf(cfg.sigmaVerticalMps)
        val (nu, H) = innovation(ekf, Rvb, rows)
        val ok = ekf.update("nhc", tS, nu, H, Mat.diag(DoubleArray(sig.size) { sig[it] * sig[it] }))
        return if (ok) "accepted" else "gated"
    }
}

object Features {
    val NAMES = listOf("acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y", "gyro_z",
        "f_norm", "f_vertical", "f_horizontal", "w_vertical")

    /** (W x 3) acc, grav, gyro vector -> (C=13 x W) float32, channel-major (the ONNX layout).
     * Computed in double and cast at the end, exactly like the reference. */
    fun make(acc: Array<DoubleArray>, grav: Array<DoubleArray>, gyro: Array<DoubleArray>, accelScale: Double, gyroScale: Double): FloatArray {
        val w = acc.size
        val out = FloatArray(13 * w)
        for (t in 0 until w) {
            val a = acc[t]
            val g = grav[t]
            val r = gyro[t]
            val gn = norm(g)
            val gHat = if (gn > 0) scale(g, 1.0 / gn) else g.copyOf()
            // the reference divides by gn (not multiplies by 1/gn): keep the same operation
            val gh = if (gn > 0) doubleArrayOf(g[0] / gn, g[1] / gn, g[2] / gn) else gHat
            val fNorm = norm(a)
            val fVert = a[0] * gh[0] + a[1] * gh[1] + a[2] * gh[2]
            val fHoriz = norm(cross(a, gh))
            val wVert = r[0] * gh[0] + r[1] * gh[1] + r[2] * gh[2]
            val ch = doubleArrayOf(a[0] / accelScale, a[1] / accelScale, a[2] / accelScale,
                g[0] / accelScale, g[1] / accelScale, g[2] / accelScale,
                r[0] / gyroScale, r[1] / gyroScale, r[2] / gyroScale,
                fNorm / accelScale - 1.0, fVert / accelScale - 1.0, fHoriz / accelScale, wVert / gyroScale)
            for (c in 0 until 13) out[c * w + t] = ch[c].toFloat()
        }
        return out
    }
}

/** A model that estimates forward speed [m/s] + variance from a phone-sample window. */
interface SpeedModel {
    val window: Int
    /** acc, grav: W x 3 (m/s^2, device frame); wz: W (rad/s, vertical gyro). */
    fun predict(acc: Array<DoubleArray>, grav: Array<DoubleArray>, wz: DoubleArray): Pair<Double, Double>
}

/** Port of navcore.fusion.ai_models.ImuWindow: restarts on a gap or non-increasing time. */
class ImuWindow(val size: Int, private val maxDtS: Double = 0.15) {
    private val t = ArrayDeque<Double>()
    private val acc = ArrayDeque<DoubleArray>()
    private val grav = ArrayDeque<DoubleArray>()
    private val wz = ArrayDeque<Double>()

    fun push(tS: Double, a: DoubleArray, g: DoubleArray, w: Double) {
        val last = t.lastOrNull()
        if (last != null && !(tS - last > 0 && tS - last <= maxDtS)) clear()
        t.addLast(tS); acc.addLast(a.copyOf()); grav.addLast(g.copyOf()); wz.addLast(w)
        if (t.size > size) {
            t.removeFirst(); acc.removeFirst(); grav.removeFirst(); wz.removeFirst()
        }
    }

    fun clear() {
        t.clear(); acc.clear(); grav.clear(); wz.clear()
    }

    fun ready(): Boolean = t.size == size

    /** The last [n] samples (oldest first). */
    fun last(n: Int): Triple<Array<DoubleArray>, Array<DoubleArray>, DoubleArray> {
        val k = acc.size - n
        return Triple(Array(n) { acc[k + it] }, Array(n) { grav[k + it] }, DoubleArray(n) { wz[k + it] })
    }
}

/** (nu, H) for an AI horizontal-speed measurement, or null when |v_h| < [minSpeedMps]. */
fun speedInnovation(ekf: ErrorStateEkf, speedMps: Double, minSpeedMps: Double = 1.0): Pair<DoubleArray, Mat>? {
    val v = ekf.nominal.vEnu
    val sh = hypot(v[0], v[1])
    if (sh < minSpeedMps) return null
    val H = Mat(1, N_STATES)
    H[0, 3] = v[0] / sh
    H[0, 4] = v[1] / sh
    return doubleArrayOf(speedMps - sh) to H
}

/** logvar clamp of navcore.fusion.ai_models (LOGVAR_CLAMP). */
fun clampedVariance(logvar: Double): Double = exp(logvar.coerceIn(-8.0, 6.0))

/** Population standard deviation (numpy np.std, ddof = 0). */
fun populationStd(x: List<Double>): Double {
    val m = x.sum() / x.size
    var s = 0.0
    for (v in x) s += (v - m) * (v - m)
    return sqrt(s / x.size)
}
