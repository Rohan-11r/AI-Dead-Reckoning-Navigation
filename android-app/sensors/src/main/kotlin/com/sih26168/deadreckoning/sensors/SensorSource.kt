package com.sih26168.deadreckoning.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import com.sih26168.deadreckoning.core.AndroidSensorType
import com.sih26168.deadreckoning.core.ChannelSample
import com.sih26168.deadreckoning.core.RawSensorEvent
import com.sih26168.deadreckoning.core.SensorEventMapper
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow

/**
 * SensorManager wrapper: accelerometer, gyroscope, magnetometer and the OS gravity estimate
 * as one cold [Flow] of [ChannelSample]s on the session clock.
 *
 * Threading: listeners are registered with a Handler on a DEDICATED HandlerThread, so
 * sensor callbacks never run on the main (UI) thread; the flow is collected by the
 * acquisition service's own coroutine. The sensor arrival rate is therefore independent of
 * the UI frame rate. Cancelling the collector unregisters the listeners and quits the thread.
 *
 * Sampling: [samplingPeriodUs] is a HINT to Android (10 ms = 100 Hz; IO-VNBD recorded 10 Hz).
 * Rates above 200 Hz would need the HIGH_SAMPLING_RATE_SENSORS permission; we do not ask for
 * them. The rate actually delivered is MEASURED downstream (core RateMeter), never assumed.
 *
 * Backpressure: an unbounded buffer would hide a stalled consumer. We use a large bounded
 * buffer (4096 samples, ~10 s at 4 x 100 Hz); when it is full, trySend FAILS and the new
 * sample is dropped and COUNTED in [SensorSourceStats.dropped], visible on the diagnostics
 * screen. (DROP_OLDEST would make trySend always succeed, and drops would be invisible.)
 */
class SensorSource(
    context: Context,
    private val sessionStartNs: Long,
    private val samplingPeriodUs: Int = 10_000,
) {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    /** Which of the wanted sensors this device has (a phone without a gyroscope exists). */
    val available: Map<String, Boolean> = WANTED.associate { (name, type) -> name to (sm.getDefaultSensor(type) != null) }

    val stats = SensorSourceStats()

    fun samples(): Flow<ChannelSample> = callbackFlow {
        val mapper = SensorEventMapper(sessionStartNs)
        val thread = HandlerThread("sih26168-sensors").apply { start() }
        val handler = Handler(thread.looper)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val s = mapper.map(RawSensorEvent(event.sensor.type, event.timestamp, event.values.copyOf(), event.accuracy))
                stats.mapped = mapper.nMapped
                stats.rejected = mapper.nRejected
                stats.lastRejection = mapper.lastRejection
                if (s != null && !trySend(s).isSuccess) stats.dropped++
            }

            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
                stats.accuracyChanges++
            }
        }
        for ((_, type) in WANTED) {
            sm.getDefaultSensor(type)?.let { sm.registerListener(listener, it, samplingPeriodUs, handler) }
        }
        awaitClose {
            sm.unregisterListener(listener)
            thread.quitSafely()
        }
    }.buffer(capacity = 4096) // SUSPEND overflow: trySend fails when full -> counted drop

    companion object {
        val WANTED = listOf(
            "accelerometer" to AndroidSensorType.ACCELEROMETER,
            "gyroscope" to AndroidSensorType.GYROSCOPE,
            "magnetometer" to AndroidSensorType.MAGNETIC_FIELD,
            "gravity" to AndroidSensorType.GRAVITY,
        )

        init {
            // the JVM-side mirror of the constants must equal the platform's
            check(AndroidSensorType.ACCELEROMETER == Sensor.TYPE_ACCELEROMETER)
            check(AndroidSensorType.GYROSCOPE == Sensor.TYPE_GYROSCOPE)
            check(AndroidSensorType.MAGNETIC_FIELD == Sensor.TYPE_MAGNETIC_FIELD)
            check(AndroidSensorType.GRAVITY == Sensor.TYPE_GRAVITY)
        }
    }
}

/** Counters written from the sensor thread, read by the diagnostics screen. */
class SensorSourceStats {
    @Volatile var mapped = 0L
    @Volatile var rejected = 0L
    @Volatile var dropped = 0L
    @Volatile var accuracyChanges = 0L
    @Volatile var lastRejection: String? = null
}
