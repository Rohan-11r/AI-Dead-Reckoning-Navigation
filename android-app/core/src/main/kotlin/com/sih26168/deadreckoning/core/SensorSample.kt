package com.sih26168.deadreckoning.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Shared sensor-sample schema: the Kotlin port of `navcore.sensors.samples` (Python reference).
 *
 * Same contract as the reference:
 *  - every sample carries `tS`, seconds on ONE monotonic session clock (Double); no nominal rate;
 *  - every vector carries its [Frame] explicitly;
 *  - SI units, named in the fields;
 *  - construction validates shape, finiteness and physical plausibility and THROWS
 *    [SensorSampleException] -- a bad sample is never silently patched (AGENTS.md 5).
 * The plausibility limits are the reference's constants, value for value.
 */

/** Frames, as `navcore.geometry.frames.Frame`. DEVICE is the phone's own axes (= the body). */
enum class Frame { DEVICE, VEHICLE, ENU, NED, ECEF, GEODETIC }

class SensorSampleException(message: String) : IllegalArgumentException(message)

/** Plausibility limits: identical to `navcore/sensors/samples.py`. */
object Limits {
    const val G0_MPS2 = 9.80665
    const val MAX_SPECIFIC_FORCE_MPS2 = 16.0 * G0_MPS2
    const val MAX_ANGULAR_RATE_RADPS = 35.0
    const val MAX_GROUND_SPEED_MPS = 100.0
    const val MAX_ABS_HEIGHT_M = 20_000.0
    /** Earth's field is 25-65 uT; allow for strong in-car disturbance, reject garbage. */
    const val MAX_MAGNETIC_FIELD_UT = 2_000.0
}

data class Vec3(val x: Double, val y: Double, val z: Double) {
    val norm: Double get() = sqrt(x * x + y * y + z * z)
    val maxAbs: Double get() = maxOf(abs(x), abs(y), abs(z))
    fun isFinite(): Boolean = x.isFinite() && y.isFinite() && z.isFinite()
    fun toList(): List<Double> = listOf(x, y, z)

    companion object {
        val ZERO = Vec3(0.0, 0.0, 0.0)
    }
}

internal fun requireTime(tS: Double, what: String = "timestamp"): Double {
    if (!tS.isFinite()) throw SensorSampleException("$what must be finite, got $tS")
    return tS
}

internal fun requireVec(name: String, v: Vec3): Vec3 {
    if (!v.isFinite()) throw SensorSampleException("$name contains non-finite values: $v")
    return v
}

/** Which physical channel a single Android sensor event belongs to. */
enum class Channel(val unit: String) {
    ACCEL("m/s^2"),    // specific force INCLUDING gravity (Android TYPE_ACCELEROMETER)
    GYRO("rad/s"),     // angular rate (TYPE_GYROSCOPE, bias-compensated by the OS)
    MAG("uT"),         // magnetic field (TYPE_MAGNETIC_FIELD)
    GRAVITY("m/s^2"),  // the OS gravity estimate (TYPE_GRAVITY; IO-VNBD's GRAVITY channel)
}

/**
 * One vector from one sensor at one instant, in [frame] (DEVICE for raw Android events).
 * Android delivers accelerometer, gyroscope and magnetometer as SEPARATE events at their own
 * times; [ImuAssembler] pairs them into an [ImuSample] with an explicit staleness bound.
 */
data class ChannelSample(
    val tS: Double,
    val channel: Channel,
    val value: Vec3,
    val frame: Frame = Frame.DEVICE,
    /** Android SENSOR_STATUS_* (0 unreliable .. 3 high); -1 when unknown. */
    val accuracy: Int = -1,
) {
    init {
        requireTime(tS)
        requireVec(channel.name, value)
        if (frame == Frame.GEODETIC || frame == Frame.ECEF) {
            throw SensorSampleException("sensor vectors are body-fixed, not $frame")
        }
        when (channel) {
            Channel.ACCEL, Channel.GRAVITY -> if (value.norm > Limits.MAX_SPECIFIC_FORCE_MPS2) {
                throw SensorSampleException("specific force ${value.norm} m/s^2 exceeds full scale")
            }
            Channel.GYRO -> if (value.maxAbs > Limits.MAX_ANGULAR_RATE_RADPS) {
                throw SensorSampleException("angular rate ${value.maxAbs} rad/s exceeds full scale")
            }
            Channel.MAG -> if (value.norm > Limits.MAX_MAGNETIC_FIELD_UT) {
                throw SensorSampleException("magnetic field ${value.norm} uT is not plausible")
            }
        }
    }
}

/**
 * Specific force [m/s^2] and angular rate [rad/s] in [frame] at one time: the reference's
 * `ImuSample`. [gyroValid] marks the axes that carry a real measurement; invalid axes must
 * be exactly 0 (never a guessed value) -- the reference rule.
 */
data class ImuSample(
    val tS: Double,
    val accelMps2: Vec3,
    val gyroRadps: Vec3,
    val frame: Frame = Frame.DEVICE,
    val gyroValid: BooleanArray = booleanArrayOf(true, true, true),
) {
    init {
        requireTime(tS)
        requireVec("accel_mps2", accelMps2)
        requireVec("gyro_radps", gyroRadps)
        if (accelMps2.norm > Limits.MAX_SPECIFIC_FORCE_MPS2) {
            throw SensorSampleException("specific force ${accelMps2.norm} m/s^2 exceeds full scale")
        }
        if (gyroRadps.maxAbs > Limits.MAX_ANGULAR_RATE_RADPS) {
            throw SensorSampleException("angular rate ${gyroRadps.maxAbs} rad/s exceeds full scale")
        }
        if (frame == Frame.GEODETIC || frame == Frame.ECEF) {
            throw SensorSampleException("IMU samples are body-fixed, not $frame")
        }
        if (gyroValid.size != 3) throw SensorSampleException("gyroValid must have 3 entries")
        val g = gyroRadps.toList()
        for (i in 0..2) {
            if (!gyroValid[i] && g[i] != 0.0) {
                throw SensorSampleException("invalid gyro axes must be zero, not a guessed value")
            }
        }
    }

    // BooleanArray has identity equality; compare by content so the data class stays a value.
    override fun equals(other: Any?): Boolean = other is ImuSample && tS == other.tS &&
        accelMps2 == other.accelMps2 && gyroRadps == other.gyroRadps && frame == other.frame &&
        gyroValid.contentEquals(other.gyroValid)

    override fun hashCode(): Int =
        listOf(tS, accelMps2, gyroRadps, frame, gyroValid.contentHashCode()).hashCode()
}

/**
 * A GNSS fix: the reference's `GnssSample`. [tS] is the EPOCH the fix describes,
 * [tReceivedS] when it became available (causality: never use a fix before it); position
 * WGS84 geodetic in RADIANS and metres; speed [m/s] and bearing [rad, clockwise from North]
 * come together or not at all.
 */
data class GnssSample(
    val tS: Double,
    val latRad: Double,
    val lonRad: Double,
    val hM: Double,
    val horizontalAccuracyM: Double,
    val tReceivedS: Double = tS,
    val speedMps: Double? = null,
    val bearingRad: Double? = null,
    val satsUsed: Int? = null,
) {
    init {
        requireTime(tS)
        requireTime(tReceivedS, "receipt time")
        if (!latRad.isFinite() || abs(latRad) > PI / 2) {
            throw SensorSampleException("latitude $latRad rad outside [-pi/2, pi/2] (degrees passed?)")
        }
        if (!lonRad.isFinite() || abs(lonRad) > PI) {
            throw SensorSampleException("longitude $lonRad rad outside [-pi, pi] (degrees passed?)")
        }
        if (!hM.isFinite() || abs(hM) > Limits.MAX_ABS_HEIGHT_M) {
            throw SensorSampleException("implausible height $hM m")
        }
        if (!(horizontalAccuracyM.isFinite() && horizontalAccuracyM > 0.0)) {
            throw SensorSampleException("horizontal accuracy must be finite and > 0, got $horizontalAccuracyM")
        }
        if (tReceivedS < tS) throw SensorSampleException("a fix cannot be received before the epoch it describes")
        if ((speedMps == null) != (bearingRad == null)) {
            throw SensorSampleException("speed and bearing come together or not at all")
        }
        if (speedMps != null && bearingRad != null) {
            if (!(speedMps.isFinite() && speedMps >= 0.0 && speedMps <= Limits.MAX_GROUND_SPEED_MPS && bearingRad.isFinite())) {
                throw SensorSampleException("implausible speed/bearing $speedMps, $bearingRad")
            }
        }
    }

    val latencyS: Double get() = tReceivedS - tS
}

/**
 * Real-dt bookkeeping for a sample stream: the reference's `TimestampGuard`, same semantics.
 * Returns dt for accepted samples and COUNTS (never hides) backwards steps, duplicates, gaps.
 */
class TimestampGuard(private val maxGapS: Double = 1.0) {
    var lastT: Double? = null
        private set
    var nAccepted = 0
        private set
    var nBackwards = 0
        private set
    var nDuplicates = 0
        private set
    var nGaps = 0
        private set

    /** dt [s] since the previous accepted sample; null = first sample, rejected, or a gap. */
    fun step(tS: Double): Double? {
        val t = requireTime(tS)
        val last = lastT
        if (last == null) {
            lastT = t
            nAccepted++
            return null
        }
        val dt = t - last
        if (dt < 0) {
            nBackwards++
            return null
        }
        if (dt == 0.0) {
            nDuplicates++
            return null
        }
        lastT = t
        nAccepted++
        if (dt > maxGapS) {
            nGaps++
            return null
        }
        return dt
    }
}
