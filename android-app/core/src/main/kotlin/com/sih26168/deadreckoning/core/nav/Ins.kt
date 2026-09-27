package com.sih26168.deadreckoning.core.nav

import kotlin.math.cos

/**
 * Port of navcore.ins.mechanization: strapdown INS in the local-level ENU frame.
 * Earth rate, transport rate, Coriolis retained; Somigliana gravity; Heun velocity and
 * trapezoidal position; attitude q_nb' = exp(-w_in dt) (x) q_nb (x) exp(w_ib dt).
 * No GNSS field is read here. Parity: nav_ins.json.
 */
class NavState(
    val tS: Double,
    val latRad: Double,
    val lonRad: Double,
    val hM: Double,
    v: DoubleArray = DoubleArray(3),
    q: DoubleArray = Quat.IDENTITY,
) {
    val vEnu: DoubleArray = v.copyOf()
    val qNb: DoubleArray = Quat.normalize(q)

    init {
        require(vEnu.size == 3 && q.size == 4) { "NavState needs v (3) and q (4)" }
        require(listOf(tS, latRad, lonRad, hM).all { it.isFinite() } && vEnu.all { it.isFinite() } && qNb.all { it.isFinite() }) {
            "NavState must be finite -- the INS has diverged or was fed NaN"
        }
    }

    fun copy(
        tS: Double = this.tS, latRad: Double = this.latRad, lonRad: Double = this.lonRad, hM: Double = this.hM,
        v: DoubleArray = vEnu, q: DoubleArray = qNb,
    ) = NavState(tS, latRad, lonRad, hM, v, q)
}

object Ins {
    private fun vdot(lat: Double, h: Double, v: DoubleArray, fN: DoubleArray): DoubleArray {
        val wIe = Geodesy.earthRateEnu(lat)
        val wEn = Geodesy.transportRateEnu(lat, h, v)
        val c = cross(add(scale(wIe, 2.0), wEn), v)
        val g = Geodesy.gravityEnu(lat, h)
        return DoubleArray(3) { fN[it] - c[it] + g[it] }
    }

    private fun positionRates(lat: Double, h: Double, v: DoubleArray): DoubleArray {
        val rm = Geodesy.meridianRadius(lat) + h
        val rn = Geodesy.primeVerticalRadius(lat) + h
        return doubleArrayOf(v[1] / rm, v[0] / (rn * cos(lat)), v[2])
    }

    /** Advance by one IMU interval; f_b [m/s^2] and w_ib [rad/s] calibrated, body frame,
     * zero-order hold on the sample at the START of the interval. */
    fun propagate(s: NavState, fB: DoubleArray, wIb: DoubleArray, dt: Double): NavState {
        require(dt > 0 && dt.isFinite()) { "dt must be finite and > 0, got $dt" }
        val lat = s.latRad
        val lon = s.lonRad
        val h = s.hM
        val v = s.vEnu
        val q = s.qNb
        val wIn = add(Geodesy.earthRateEnu(lat), Geodesy.transportRateEnu(lat, h, v))
        val qNew = Quat.normalize(Quat.mul(Quat.fromRotvec(scale(wIn, -dt)), Quat.mul(q, Quat.fromRotvec(scale(wIb, dt)))))

        val fN = Quat.rotate(Quat.slerp(q, qNew, 0.5), fB)
        val a0 = vdot(lat, h, v, fN)
        val vPred = DoubleArray(3) { v[it] + a0[it] * dt }
        val r0 = positionRates(lat, h, DoubleArray(3) { 0.5 * (v[it] + vPred[it]) })
        val latP = lat + r0[0] * dt
        val hP = h + r0[2] * dt
        val a1 = vdot(latP, hP, vPred, fN)
        val vNew = DoubleArray(3) { v[it] + 0.5 * (a0[it] + a1[it]) * dt }

        val vAvg = DoubleArray(3) { 0.5 * (v[it] + vNew[it]) }
        val latMid = lat + 0.5 * positionRates(lat, h, vAvg)[0] * dt
        val hMid = h + 0.5 * vAvg[2] * dt
        val r = positionRates(latMid, hMid, vAvg)
        val lonNew = Geodesy.wrapPi(lon + r[1] * dt)
        return NavState(s.tS + dt, lat + r[0] * dt, lonNew, h + r[2] * dt, vNew, qNew)
    }
}
