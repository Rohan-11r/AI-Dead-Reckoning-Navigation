package com.sih26168.deadreckoning.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Synthetic accelerometer input, built in each test; analytically known outcomes. */
class AiDiagnosticsTest {
    private val g = 9.80665

    /** 100 Hz samples of a still phone (z = g) from [t0] for [seconds]; [spikeAt] adds [spike] to z for one sample. */
    private fun feed(m: ShockMonitor, t0: Double, seconds: Double, spikeAt: Double? = null, spike: Double = 0.0) {
        val n = (seconds * 100).toInt()
        for (i in 0 until n) {
            val t = t0 + i * 0.01 + 0.005 // off the bin edges
            val z = if (spikeAt != null && kotlin.math.abs(t - spikeAt) < 0.004) g + spike else g
            m.offer(t, 0.0, 0.0, z)
        }
    }

    @Test
    fun stillPhoneHasNoShocks() {
        val m = ShockMonitor()
        feed(m, 0.0, 5.0)
        assertEquals(0L, m.nShocks)
        assertTrue(m.nBins >= 49)
    }

    @Test
    fun singleSampleSpikeIsDetectedAndAveragedDown() {
        val m = ShockMonitor(thresholdMps2 = 4.0)
        feed(m, 0.0, 2.0)
        feed(m, 2.0, 1.0, spikeAt = 2.505, spike = 20.0)
        assertNotNull(m.last)
        val e = m.last!!
        assertEquals(1L, m.nShocks)
        assertEquals(20.0, e.rawPeakDevMps2, 1e-9)
        // 10 samples in the bin, one of them +20: the bin mean is +2 -> below the threshold
        assertEquals(2.0, e.averagedDevMps2, 1e-9)
        assertTrue(e.suppressed)
        assertEquals(2.6, e.tS, 1e-9) // bin (2.5, 2.6]
    }

    @Test
    fun nothingIsDetectedDuringWarmup() {
        val m = ShockMonitor(warmupBins = 10)
        feed(m, 0.0, 0.5, spikeAt = 0.205, spike = 30.0)
        assertEquals(0L, m.nShocks)
    }

    @Test
    fun flashHoldsForTheConfiguredTime() {
        val e = ShockEvent(10.0, 20.0, 2.0, true)
        assertTrue(VibrationStatus(true, 1, 1, e, nowS = 11.0).flashing(holdS = 1.5))
        assertFalse(VibrationStatus(true, 1, 1, e, nowS = 12.0).flashing(holdS = 1.5))
        assertFalse(VibrationStatus(true, 1, 0, null, nowS = 12.0).flashing())
    }

    @Test
    fun nhcFollowsTheCounterThatGrew() {
        val m = NhcMonitor()
        assertEquals(NhcStatus.Phase.WAITING, m.update(null).phase)
        assertEquals(NhcStatus.Phase.APPLIED, m.update(mapOf("nhc_accepted" to 3L)).phase)
        assertEquals(NhcStatus.Phase.LOW_SPEED, m.update(mapOf("nhc_accepted" to 3L, "nhc_low_speed" to 1L)).phase)
        // no growth: keep the last outcome
        assertEquals(NhcStatus.Phase.LOW_SPEED, m.update(mapOf("nhc_accepted" to 3L, "nhc_low_speed" to 1L)).phase)
        val s = m.update(mapOf("nhc_accepted" to 4L, "nhc_low_speed" to 1L))
        assertEquals(NhcStatus.Phase.APPLIED, s.phase)
        assertEquals(4L, s.acceptedTotal)
    }

    @Test
    fun nhcCountsANewNavigatorFromZero() {
        val m = NhcMonitor()
        m.update(mapOf("nhc_accepted" to 100L))
        assertEquals(NhcStatus.Phase.GATED, m.update(mapOf("nhc_gated" to 1L)).phase)
    }
}
