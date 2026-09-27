package com.sih26168.deadreckoning.core

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Locale

class DisplayTest {
    @Test
    fun `speed is shown in whole km per hour and missing speed is not zero`() {
        assertEquals("54 km/h", Display.speedKmh(15.0))
        assertEquals("0 km/h", Display.speedKmh(0.0))
        assertEquals(Display.NA, Display.speedKmh(null))
        assertEquals(Display.NA, Display.speedKmh(Double.NaN))
    }

    @Test
    fun `confidence percent is clamped and missing is not zero`() {
        assertEquals("39 %", Display.percent(0.3935))
        assertEquals("100 %", Display.percent(1.2))
        assertEquals(Display.NA, Display.percent(null))
    }

    @Test
    fun `lost is shown as dead reckoning and a simulated outage is always labelled`() {
        assertEquals("DEAD_RECKONING", Display.stateLabel(GnssState.LOST, false))
        assertEquals("GOOD (SIMULATED OUTAGE)", Display.stateLabel(GnssState.GOOD, true))
        assertEquals("STARTING", Display.stateLabel(null, false))
    }

    @Test
    fun `numbers are locale independent`() {
        val old = Locale.getDefault()
        Locale.setDefault(Locale.FRANCE)
        try {
            assertEquals("12.5 m", Display.metres(12.5))
            assertEquals("+0.100  -0.200  +9.810", Display.vec(Vec3(0.1, -0.2, 9.81)))
            assertEquals("100.0 Hz", Display.hz(100.0))
        } finally {
            Locale.setDefault(old)
        }
    }

    @Test
    fun `road name says why it is missing`() {
        assertEquals("— (no offline map bundled)", Display.roadName(null))
        assertEquals("— (no confident road match)", Display.roadName(null, hasMap = true))
        assertEquals("High Street", Display.roadName("High Street", hasMap = true))
    }
}
