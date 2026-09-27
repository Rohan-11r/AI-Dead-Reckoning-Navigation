package com.sih26168.deadreckoning.core.nav

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cbrt
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Port of navcore.common.constants, navcore.geometry.quaternion and navcore.geometry.geodesy.
 * Same conventions (docs/navigation_math.md sec 4.1): q = [w, x, y, z], Hamilton product,
 * q_ab represents R_b^a, canonical w >= 0 for created quaternions. Parity: nav_geometry.json.
 */
object Const {
    const val WGS84_A_M = 6378137.0
    const val WGS84_F = 1.0 / 298.257223563
    const val WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
    const val WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
    const val WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)
    const val WGS84_OMEGA_IE_RADPS = 7.292115e-5
    const val WGS84_GAMMA_E_MPS2 = 9.7803253359
    const val WGS84_SOMIGLIANA_K = 1.93185265241e-3
    const val FREE_AIR_GRADIENT_PER_S2 = 3.086e-6
    const val STANDARD_GRAVITY_MPS2 = 9.80665
    /** Below this the quaternion exponential uses its Taylor series (a parity-relevant constant). */
    const val SMALL_ANGLE_THRESHOLD_RAD = 1e-6
}

object Quat {
    val IDENTITY: DoubleArray get() = doubleArrayOf(1.0, 0.0, 0.0, 0.0)

    fun normalize(q: DoubleArray): DoubleArray {
        val n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        require(n != 0.0) { "cannot normalise a zero quaternion" }
        return doubleArrayOf(q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    }

    fun canonical(q: DoubleArray): DoubleArray = if (q[0] < 0.0) DoubleArray(4) { -q[it] } else q.copyOf()

    fun conj(q: DoubleArray) = doubleArrayOf(q[0], -q[1], -q[2], -q[3])

    /** Hamilton product p (x) q: apply q first, then p. */
    fun mul(p: DoubleArray, q: DoubleArray) = doubleArrayOf(
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    )

    /** R(q) v via v + 2w (u x v) + 2 u x (u x v). */
    fun rotate(q: DoubleArray, v: DoubleArray): DoubleArray {
        val u = doubleArrayOf(q[1], q[2], q[3])
        val t = scale(cross(u, v), 2.0)
        val c = cross(u, t)
        return doubleArrayOf(v[0] + q[0] * t[0] + c[0], v[1] + q[0] * t[1] + c[1], v[2] + q[0] * t[2] + c[2])
    }

    fun toDcm(q0: DoubleArray): Mat {
        val q = normalize(q0)
        val (w, x, y, z) = q.toList()
        return Mat.of(3, 3,
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
    }

    /** Shepperd's method; canonical w >= 0. */
    fun fromDcm(m: Mat): DoubleArray {
        val tr = m[0, 0] + m[1, 1] + m[2, 2]
        val cand = doubleArrayOf(tr, m[0, 0], m[1, 1], m[2, 2])
        var k = 0
        for (i in 1..3) if (cand[i] > cand[k]) k = i // numpy argmax: first maximum wins
        val q = when (k) {
            0 -> { val s = 2.0 * sqrt(1.0 + tr); doubleArrayOf(0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s) }
            1 -> { val s = 2.0 * sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]); doubleArrayOf((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s) }
            2 -> { val s = 2.0 * sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]); doubleArrayOf((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s) }
            else -> { val s = 2.0 * sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]); doubleArrayOf((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s) }
        }
        return canonical(normalize(q))
    }

    /** Exponential map with the reference's small-angle Taylor branch. */
    fun fromRotvec(rv: DoubleArray): DoubleArray {
        val theta = norm(rv)
        val small = theta < Const.SMALL_ANGLE_THRESHOLD_RAD
        val halfSinc = if (small) 0.5 - theta * theta / 48.0 else sin(0.5 * theta) / theta
        val w = if (small) 1.0 - theta * theta / 8.0 else cos(0.5 * theta)
        return normalize(doubleArrayOf(w, halfSinc * rv[0], halfSinc * rv[1], halfSinc * rv[2]))
    }

    fun slerp(q0in: DoubleArray, q1in: DoubleArray, t: Double): DoubleArray {
        val q0 = normalize(q0in)
        var q1 = normalize(q1in)
        var d = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
        if (d < 0.0) q1 = DoubleArray(4) { -q1[it] }
        d = abs(d)
        val near = d > 1.0 - 1e-12
        val omega = acos(d.coerceIn(-1.0, 1.0))
        val so = if (near) 1.0 else sin(omega)
        val a = if (near) 1.0 - t else sin((1.0 - t) * omega) / so
        val b = if (near) t else sin(t * omega) / so
        return normalize(DoubleArray(4) { a * q0[it] + b * q1[it] })
    }

    fun fromEuler(yaw: Double, pitch: Double, roll: Double): DoubleArray {
        val cy = cos(0.5 * yaw); val sy = sin(0.5 * yaw)
        val cp = cos(0.5 * pitch); val sp = sin(0.5 * pitch)
        val cr = cos(0.5 * roll); val sr = sin(0.5 * roll)
        return canonical(normalize(doubleArrayOf(
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        )))
    }
}

object Geodesy {
    fun meridianRadius(lat: Double): Double {
        val s2 = sin(lat) * sin(lat)
        return Const.WGS84_A_M * (1.0 - Const.WGS84_E2) / (1.0 - Const.WGS84_E2 * s2).pow(1.5)
    }

    fun primeVerticalRadius(lat: Double): Double = Const.WGS84_A_M / sqrt(1.0 - Const.WGS84_E2 * sin(lat) * sin(lat))

    /** Somigliana normal gravity with free-air correction [m/s^2]. */
    fun normalGravity(lat: Double, h: Double): Double {
        val s2 = sin(lat) * sin(lat)
        return Const.WGS84_GAMMA_E_MPS2 * (1.0 + Const.WGS84_SOMIGLIANA_K * s2) / sqrt(1.0 - Const.WGS84_E2 * s2) -
            Const.FREE_AIR_GRADIENT_PER_S2 * h
    }

    fun geodeticToEcef(lat: Double, lon: Double, h: Double): DoubleArray {
        val n = primeVerticalRadius(lat)
        val cl = cos(lat)
        return doubleArrayOf((n + h) * cl * cos(lon), (n + h) * cl * sin(lon), (n * (1.0 - Const.WGS84_E2) + h) * sin(lat))
    }

    /** ECEF -> (lat, lon, h), Heikkinen closed form (as the reference). */
    fun ecefToGeodetic(p: DoubleArray): DoubleArray {
        val (x, y, z) = p.toList()
        val a = Const.WGS84_A_M
        val b = Const.WGS84_B_M
        val e2 = Const.WGS84_E2
        val ep2 = Const.WGS84_EP2
        val pp = hypot(x, y)
        require(hypot(pp, z) >= 50_000.0) { "ecefToGeodetic: point too close to the Earth's centre" }
        val F = 54.0 * b * b * z * z
        val G = pp * pp + (1.0 - e2) * z * z - e2 * (a * a - b * b)
        val c = e2 * e2 * F * pp * pp / (G * G * G)
        val s = cbrt(1.0 + c + sqrt(c * c + 2.0 * c))
        val k = s + 1.0 + 1.0 / s
        val P = F / (3.0 * k * k * G * G)
        val Q = sqrt(1.0 + 2.0 * e2 * e2 * P)
        val r0 = -(P * e2 * pp) / (1.0 + Q) +
            sqrt(max(0.5 * a * a * (1.0 + 1.0 / Q) - P * (1.0 - e2) * z * z / (Q * (1.0 + Q)) - 0.5 * P * pp * pp, 0.0))
        val U = hypot(pp - e2 * r0, z)
        val V = sqrt((pp - e2 * r0) * (pp - e2 * r0) + (1.0 - e2) * z * z)
        val z0 = b * b * z / (a * V)
        val h = U * (1.0 - b * b / (a * V))
        return doubleArrayOf(atan2(z + ep2 * z0, pp), atan2(y, x), h)
    }

    /** R_e^n (ENU rows). */
    fun rEcefToEnu(lat: Double, lon: Double): Mat {
        val sl = sin(lat); val cl = cos(lat); val so = sin(lon); val co = cos(lon)
        return Mat.of(3, 3, -so, co, 0.0, -sl * co, -sl * so, cl, cl * co, cl * so, sl)
    }

    fun earthRateEnu(lat: Double) = doubleArrayOf(0.0, Const.WGS84_OMEGA_IE_RADPS * cos(lat), Const.WGS84_OMEGA_IE_RADPS * sin(lat))

    fun transportRateEnu(lat: Double, h: Double, v: DoubleArray): DoubleArray {
        val rm = meridianRadius(lat) + h
        val rn = primeVerticalRadius(lat) + h
        return doubleArrayOf(-v[1] / rm, v[0] / rn, v[0] * tan(lat) / rn)
    }

    fun gravityEnu(lat: Double, h: Double) = doubleArrayOf(0.0, 0.0, -normalGravity(lat, h))

    /** math.remainder(x, 2 pi): IEEE remainder (round-half-even quotient), as Python. */
    fun wrapPi(x: Double): Double = Math.IEEEremainder(x, 2.0 * PI)
}

/** ENU frame anchored at a fixed origin (navcore.geometry.geodesy.LocalTangentPlane). */
class LocalTangentPlane(val lat0: Double, val lon0: Double, val h0: Double) {
    private val origin = Geodesy.geodeticToEcef(lat0, lon0, h0)
    private val r = Geodesy.rEcefToEnu(lat0, lon0)

    fun geodeticToEnu(lat: Double, lon: Double, h: Double): DoubleArray =
        r.mulVec(sub(Geodesy.geodeticToEcef(lat, lon, h), origin))

    fun enuToGeodetic(enu: DoubleArray): DoubleArray = Geodesy.ecefToGeodetic(add(r.t().mulVec(enu), origin))
}
