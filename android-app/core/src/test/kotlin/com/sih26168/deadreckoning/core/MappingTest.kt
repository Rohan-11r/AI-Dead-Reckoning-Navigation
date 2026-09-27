package com.sih26168.deadreckoning.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI

/** Android SensorEvent / Location -> shared schema. Inputs are SYNTHETIC with known answers. */
class MappingTest {
    private val start = 5_000_000_000L // session start on the elapsed-realtime clock [ns]

    private fun ev(type: Int, ns: Long, vararg v: Float, acc: Int = 3) = RawSensorEvent(type, ns, v, acc)

    @Test
    fun `each sensor type maps to its channel on the session clock in the device frame`() {
        val m = SensorEventMapper(start)
        val a = m.map(ev(AndroidSensorType.ACCELEROMETER, start + 1_500_000_000L, 0.1f, -0.2f, 9.8f))!!
        assertEquals(Channel.ACCEL, a.channel)
        assertEquals(1.5, a.tS, 1e-12)
        assertEquals(Frame.DEVICE, a.frame)
        assertEquals(9.8, a.value.z, 1e-6)
        assertEquals(Channel.GYRO, m.map(ev(AndroidSensorType.GYROSCOPE, start, 0.01f, 0f, -0.3f))!!.channel)
        assertEquals(Channel.MAG, m.map(ev(AndroidSensorType.MAGNETIC_FIELD, start, 20f, -5f, 40f))!!.channel)
        assertEquals(Channel.GRAVITY, m.map(ev(AndroidSensorType.GRAVITY, start, 0f, 0f, 9.81f))!!.channel)
        assertEquals(4L, m.nMapped)
    }

    @Test
    fun `axes are copied, never remapped or rescaled`() {
        val s = SensorEventMapper(start).map(ev(AndroidSensorType.GYROSCOPE, start, 0.25f, -0.5f, 0.125f))!!
        assertEquals(Vec3(0.25, -0.5, 0.125), s.value) // exactly representable floats
    }

    @Test
    fun `invalid events are rejected and counted, never patched`() {
        val m = SensorEventMapper(start)
        assertNull(m.map(ev(AndroidSensorType.ACCELEROMETER, start, Float.NaN, 0f, 9.8f)))
        assertNull(m.map(ev(AndroidSensorType.ACCELEROMETER, start, 0f, 0f, 200f))) // > 16 g
        assertNull(m.map(ev(AndroidSensorType.GYROSCOPE, start, 40f, 0f, 0f)))       // > 35 rad/s
        assertNull(m.map(ev(AndroidSensorType.ACCELEROMETER, start, 1f, 2f)))        // 2 values
        assertEquals(4L, m.nRejected)
        assertNotNull(m.lastRejection)
        assertNull(m.map(ev(18 /* STEP_DETECTOR */, start, 1f, 0f, 0f)))
        assertEquals(1L, m.nIgnoredType)
    }

    private fun loc(
        acc: Double? = 4.0, speed: Double? = 12.0, bearing: Double? = 90.0, epochNs: Long = start + 10_000_000_000L,
    ) = RawLocation("gps", 52.4, -1.5, 100.0, acc, speed, bearing, epochNs, 1_600_000_000_000L, 9)

    @Test
    fun `location maps degrees to radians with epoch and receipt on the session clock`() {
        val r = mapLocation(loc(), start, start + 10_250_000_000L) as LocationMapping.Ok
        val f = r.fix
        assertEquals(52.4 * PI / 180, f.latRad, 1e-15)
        assertEquals(-1.5 * PI / 180, f.lonRad, 1e-15)
        assertEquals(10.0, f.tS, 1e-12)
        assertEquals(10.25, f.tReceivedS, 1e-12)
        assertEquals(0.25, f.latencyS, 1e-12)
        assertEquals(PI / 2, f.bearingRad!!, 1e-15)
        assertEquals(9, f.satsUsed)
    }

    @Test
    fun `a fix without reported accuracy is rejected, not given an invented one`() {
        assertTrue(mapLocation(loc(acc = null), start, start + 10_000_000_000L) is LocationMapping.Rejected)
        assertTrue(mapLocation(loc(acc = 0.0), start, start + 10_000_000_000L) is LocationMapping.Rejected)
    }

    @Test
    fun `speed without bearing drops both`() {
        val f = (mapLocation(loc(bearing = null), start, start + 10_000_000_000L) as LocationMapping.Ok).fix
        assertNull(f.speedMps)
        assertNull(f.bearingRad)
    }

    @Test
    fun `receipt before the epoch is rejected beyond clock jitter`() {
        val ok = mapLocation(loc(), start, start + 10_000_000_000L - 500_000L) // 0.5 ms early: jitter
        assertTrue(ok is LocationMapping.Ok)
        assertEquals((ok as LocationMapping.Ok).fix.tS, ok.fix.tReceivedS, 0.0)
        assertTrue(mapLocation(loc(), start, start + 9_000_000_000L) is LocationMapping.Rejected)
    }

    @Test
    fun `imu assembler pairs accel with a fresh gyro and marks a stale one invalid`() {
        val asm = ImuAssembler(maxGyroAgeS = 0.05)
        assertNull(asm.offer(ChannelSample(1.00, Channel.GYRO, Vec3(0.0, 0.0, 0.2))))
        val a = asm.offer(ChannelSample(1.02, Channel.ACCEL, Vec3(0.0, 0.0, 9.8)))!!
        assertEquals(0.2, a.gyroRadps.z, 0.0)
        assertTrue(a.gyroValid.all { it })
        val stale = asm.offer(ChannelSample(1.10, Channel.ACCEL, Vec3(0.0, 0.0, 9.8)))!!
        assertFalse(stale.gyroValid.any { it })
        assertEquals(Vec3.ZERO, stale.gyroRadps)
        assertEquals(1L, asm.nWithoutGyro)
        assertNull(asm.offer(ChannelSample(1.2, Channel.MAG, Vec3(20.0, 0.0, 40.0))))
    }
}
