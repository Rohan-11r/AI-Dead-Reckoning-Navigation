package com.sih26168.deadreckoning.core

import kotlin.math.PI

/**
 * Raw Android sensor/location data -> the shared schema. Pure functions over plain values
 * (no android.* imports), so every rule is unit-testable on a plain JVM. The :sensors and
 * :gnss modules only copy fields out of SensorEvent / Location and call these.
 *
 * CLOCK: everything is placed on ONE session clock, SystemClock.elapsedRealtimeNanos():
 *  - SensorEvent.timestamp is nanoseconds on that clock (documented since API 17 for the
 *    common case; see RawSensorEvent.timestampNs);
 *  - Location.getElapsedRealtimeNanos() is the fix EPOCH on the same clock;
 *  - the receipt time is SystemClock.elapsedRealtimeNanos() read in the callback.
 *  t_s = (ns - sessionStartNs) / 1e9. Wall-clock (UTC) time is logged separately for joins,
 *  never used for dt (it can jump).
 */

/** Android `Sensor.TYPE_*` values (stable public API constants), mirrored for the JVM. */
object AndroidSensorType {
    const val ACCELEROMETER = 1
    const val MAGNETIC_FIELD = 2
    const val GYROSCOPE = 4
    const val GRAVITY = 9
}

/** The fields of an android.hardware.SensorEvent that the mapping uses. */
class RawSensorEvent(
    val type: Int,
    /** SensorEvent.timestamp: ns on SystemClock.elapsedRealtimeNanos(). */
    val timestampNs: Long,
    val values: FloatArray,
    val accuracy: Int,
)

class SensorEventMapper(private val sessionStartNs: Long) {
    var nMapped = 0L
        private set
    var nRejected = 0L
        private set
    var nIgnoredType = 0L
        private set
    var lastRejection: String? = null
        private set

    /** Map one event, or null if its type is not ours or it fails validation (counted). */
    fun map(e: RawSensorEvent): ChannelSample? {
        val channel = when (e.type) {
            AndroidSensorType.ACCELEROMETER -> Channel.ACCEL
            AndroidSensorType.GYROSCOPE -> Channel.GYRO
            AndroidSensorType.MAGNETIC_FIELD -> Channel.MAG
            AndroidSensorType.GRAVITY -> Channel.GRAVITY
            else -> {
                nIgnoredType++
                return null
            }
        }
        return try {
            if (e.values.size < 3) throw SensorSampleException("${channel.name} event has ${e.values.size} values")
            // Android device axes (x right, y up, z out of the screen) ARE the DEVICE frame:
            // no remapping here -- device -> vehicle alignment is the navigation core's job.
            val v = Vec3(e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble())
            ChannelSample(nsToS(e.timestampNs, sessionStartNs), channel, v, Frame.DEVICE, e.accuracy)
                .also { nMapped++ }
        } catch (x: SensorSampleException) {
            nRejected++
            lastRejection = x.message
            null
        }
    }
}

fun nsToS(ns: Long, sessionStartNs: Long): Double = (ns - sessionStartNs) / 1e9

/** The fields of an android.location.Location that the mapping uses. */
data class RawLocation(
    val provider: String,
    val latitudeDeg: Double,
    val longitudeDeg: Double,
    val altitudeM: Double?,        // null when !hasAltitude() (WGS84 ellipsoidal height on Android)
    val accuracyM: Double?,        // null when !hasAccuracy(); a 68 % horizontal radius
    val speedMps: Double?,         // null when !hasSpeed()
    val bearingDeg: Double?,       // null when !hasBearing(); degrees clockwise from North
    val elapsedRealtimeNs: Long,   // the fix EPOCH on the session clock
    val utcTimeMs: Long,           // Location.getTime(): for logs/joins only
    val satellitesUsed: Int? = null,
)

sealed interface LocationMapping {
    data class Ok(val fix: GnssSample) : LocationMapping
    data class Rejected(val reason: String) : LocationMapping
}

/**
 * Location -> [GnssSample]. A fix WITHOUT a reported accuracy is REJECTED, not given an
 * invented one (the filter weights by it). A fix without altitude gets h = 0 flagged by
 * [GnssSample.hM] == 0 and the caller's provider tag -- the navigation core does not use
 * GNSS height as a measurement of note (vertical accuracy is not reported by phones).
 * Speed and bearing are kept only together (the reference rule); a speed without a
 * bearing (common when stopped) drops both.
 */
fun mapLocation(raw: RawLocation, sessionStartNs: Long, receivedNs: Long): LocationMapping {
    val acc = raw.accuracyM ?: return LocationMapping.Rejected("no horizontal accuracy reported")
    val epoch = nsToS(raw.elapsedRealtimeNs, sessionStartNs)
    val received = nsToS(receivedNs, sessionStartNs)
    if (received < epoch) {
        // clocks on one device can disagree by a few microseconds; anything more is a bug
        if (epoch - received > 1e-3) return LocationMapping.Rejected("received ${epoch - received} s before its epoch")
    }
    val both = raw.speedMps != null && raw.bearingDeg != null
    return try {
        LocationMapping.Ok(
            GnssSample(
                tS = epoch,
                latRad = raw.latitudeDeg * PI / 180.0,
                lonRad = raw.longitudeDeg * PI / 180.0,
                hM = raw.altitudeM ?: 0.0,
                horizontalAccuracyM = acc,
                tReceivedS = maxOf(received, epoch),
                speedMps = if (both) raw.speedMps else null,
                bearingRad = if (both) raw.bearingDeg!! * PI / 180.0 else null,
                satsUsed = raw.satellitesUsed,
            ),
        )
    } catch (x: SensorSampleException) {
        LocationMapping.Rejected(x.message ?: "invalid fix")
    }
}

/**
 * Pairs the separately-delivered ACCEL and GYRO channel samples into [ImuSample]s at the
 * ACCELEROMETER's timestamps, using the latest gyro reading if it is at most
 * [maxGyroAgeS] old (zero-order hold). With no fresh gyro the sample is emitted with the
 * gyro marked INVALID and zeroed -- never a guessed value -- and counted.
 */
class ImuAssembler(private val maxGyroAgeS: Double = 0.05) {
    private var lastGyro: ChannelSample? = null
    var nAssembled = 0L
        private set
    var nWithoutGyro = 0L
        private set

    fun offer(s: ChannelSample): ImuSample? = when (s.channel) {
        Channel.GYRO -> {
            lastGyro = s
            null
        }
        Channel.ACCEL -> {
            val g = lastGyro
            val fresh = g != null && s.tS - g.tS in 0.0..maxGyroAgeS
            nAssembled++
            if (fresh) {
                ImuSample(s.tS, s.value, g!!.value, s.frame)
            } else {
                nWithoutGyro++
                ImuSample(s.tS, s.value, Vec3.ZERO, s.frame, booleanArrayOf(false, false, false))
            }
        }
        else -> null
    }
}
