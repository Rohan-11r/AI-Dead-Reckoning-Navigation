package com.sih26168.deadreckoning.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.util.Locale
import kotlin.math.exp

/** Schema, timestamps, diagnostics, outage switch, confidence, logger, placeholder engine.
 * All inputs SYNTHETIC with known answers. */
class CoreTest {
    @get:Rule
    val tmp = TemporaryFolder()

    // ---- schema ---------------------------------------------------------------------------

    @Test
    fun `imu sample invalid gyro axes must be zero`() {
        assertThrows(SensorSampleException::class.java) {
            ImuSample(0.0, Vec3(0.0, 0.0, 9.8), Vec3(0.1, 0.0, 0.0), gyroValid = booleanArrayOf(false, true, true))
        }
        ImuSample(0.0, Vec3(0.0, 0.0, 9.8), Vec3(0.0, 0.0, 0.1), gyroValid = booleanArrayOf(false, false, true))
    }

    @Test
    fun `imu samples compare by value`() {
        val a = ImuSample(1.0, Vec3(0.0, 0.0, 9.8), Vec3.ZERO)
        assertEquals(a, ImuSample(1.0, Vec3(0.0, 0.0, 9.8), Vec3.ZERO))
        assertEquals(a.hashCode(), ImuSample(1.0, Vec3(0.0, 0.0, 9.8), Vec3.ZERO).hashCode())
    }

    @Test
    fun `sensor vectors are never geodetic`() {
        assertThrows(SensorSampleException::class.java) {
            ChannelSample(0.0, Channel.ACCEL, Vec3(0.0, 0.0, 9.8), Frame.GEODETIC)
        }
    }

    @Test
    fun `timestamp guard counts backwards, duplicates and gaps like the reference`() {
        val g = TimestampGuard(maxGapS = 1.0)
        assertNull(g.step(0.0))
        assertEquals(0.1, g.step(0.1)!!, 1e-12)
        assertNull(g.step(0.05))  // backwards
        assertNull(g.step(0.1))   // duplicate
        assertNull(g.step(2.0))   // gap: restart integration
        assertEquals(0.1, g.step(2.1)!!, 1e-12)
        assertEquals(listOf(1, 1, 1, 4), listOf(g.nBackwards, g.nDuplicates, g.nGaps, g.nAccepted))
    }

    // ---- diagnostics ----------------------------------------------------------------------

    @Test
    fun `rate meter measures the stream's own rate and gaps`() {
        val r = RateMeter(windowS = 1.0)
        for (k in 0..200) r.record(k * 0.01)  // 100 Hz for 2 s
        assertEquals(100.0, r.rateHz, 1e-6)
        assertEquals(0.01, r.medianDtS!!, 1e-9)
        r.record(3.0) // a 1 s gap
        assertEquals(1L, r.nGaps)
        assertEquals(202L, r.nTotal)
    }

    @Test
    fun `simulated outage withholds fixes by epoch while on, and records its window`() {
        val o = SimulatedOutage()
        fun fix(t: Double) = GnssSample(t, 0.9, -0.03, 100.0, 4.0)
        assertFalse(o.withholds(fix(1.0)))
        o.set(true, 10.0)
        assertFalse(o.withholds(fix(9.5)))  // epoch before the toggle: not withheld
        assertTrue(o.withholds(fix(10.0)))
        assertTrue(o.withholds(fix(25.0)))
        o.set(false, 30.0)
        assertFalse(o.withholds(fix(31.0)))
        assertEquals(listOf(10.0 to 30.0), o.windows)
        assertEquals(2L, o.nWithheld)
    }

    @Test
    fun `confidence is the probability of being within 10 m`() {
        assertEquals(1.0 - exp(-0.5), positionConfidence(10.0)!!, 1e-12)  // R = sigma -> 39.3 %
        assertTrue(positionConfidence(1.0)!! > 0.999)
        assertTrue(positionConfidence(200.0)!! < 0.002)
        assertNull(positionConfidence(null))
        assertNull(positionConfidence(0.0))
        // Android accuracy is a 68 % radius: a sigma from it puts 68 % inside that radius
        val sigma = sigmaFromAndroidAccuracy(6.0)
        assertEquals(0.68, positionConfidence(sigma, radiusM = 6.0)!!, 1e-9)
    }

    // ---- logger ---------------------------------------------------------------------------

    @Test
    fun `session logger writes csv and manifest, locale-independently`() {
        val old = Locale.getDefault()
        Locale.setDefault(Locale.GERMANY) // decimal comma locale: numbers must still use '.'
        try {
            val dir = tmp.newFolder("s1")
            SessionLogger(dir, mapOf("device" to "test \"phone\"", "session_start_utc_ms" to 1L), flushEveryLines = 2).use { log ->
                log.sensor(ChannelSample(0.01, Channel.ACCEL, Vec3(0.1, -0.2, 9.81), accuracy = 3))
                log.fix(GnssSample(1.0, 0.9145525280450286, -0.02617993877991494, 100.0, 4.5, 1.25, 12.0, 0.5), "gps", true)
                log.event(1.25, "state", "LOST -> RECOVERING, first fix")
            }
            val sensors = dir.resolve("sensors.csv").readLines()
            assertEquals(SessionLogger.SENSORS_HEADER.joinToString(","), sensors[0])
            assertEquals("0.01,ACCEL,0.1,-0.2,9.81,3", sensors[1])
            val gnss = dir.resolve("gnss.csv").readLines()[1].split(",")
            assertEquals(52.4, gnss[2].toDouble(), 1e-12)
            assertEquals("1", gnss.last())
            assertEquals("1.25,state,\"LOST -> RECOVERING, first fix\"", dir.resolve("events.csv").readLines()[1])
            val manifest = dir.resolve("manifest.json").readText()
            assertTrue(manifest.contains("\"closed\":true"))
            assertTrue(manifest.contains("\"device\":\"test \\\"phone\\\"\""))
            assertTrue(manifest.contains("\"sensor_lines\":1"))
        } finally {
            Locale.setDefault(old)
        }
    }

    @Test
    fun `logging after close fails loudly instead of losing data`() {
        val log = SessionLogger(tmp.newFolder("s2"), emptyMap())
        log.close()
        assertThrows(IllegalStateException::class.java) { log.event(0.0, "late") }
    }

    // ---- placeholder engine ---------------------------------------------------------------

    @Test
    fun `gnss-only engine reports no position when lost, instead of a stale one`() {
        val e = GnssOnlyEngine()
        val fix = GnssSample(1.0, 0.9145, -0.0262, 100.0, 4.0, 1.0, 10.0, 0.0)
        e.onFix(fix)
        assertEquals(GnssState.RECOVERING, e.snapshot().state)
        assertNotNull(e.snapshot().latRad)
        assertEquals(10.0, e.snapshot().speedMps!!, 0.0)
        e.onImu(ImuSample(40.0, Vec3(0.0, 0.0, 9.8), Vec3.ZERO)) // 39 s without a fix
        val s = e.snapshot()
        assertEquals(GnssState.LOST, s.state)
        assertNull(s.latRad)
        assertNull(s.confidence)
        assertNotNull(s.note)
        assertNull(s.roadName)
    }
}
