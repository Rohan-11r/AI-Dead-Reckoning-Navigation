package com.sih26168.deadreckoning.app

import org.junit.Assert.assertEquals
import org.junit.Test

class AlignmentProgressTest {
    private fun of(mode: String?, tS: Double, tInit: Double? = 10.0) = AlignmentStatus.of(mode, tS, tInit, windowS = 120.0)

    @Test
    fun waitingIsZero() {
        assertEquals(AlignmentStatus(AlignmentStatus.Phase.WAITING_FOR_GNSS, 0), of("WAITING", 0.0, null))
    }

    @Test
    fun aligningIsElapsedOverWindow() {
        assertEquals(0, of("ALIGNING", 10.0).percent)
        assertEquals(50, of("ALIGNING", 70.0).percent)   // 60 s of 120 s
        assertEquals(AlignmentStatus.Phase.ALIGNING, of("ALIGNING", 70.0).phase)
    }

    @Test
    fun aligningNeverClaimsLockedBeforeTheEngineSwitches() {
        assertEquals(99, of("ALIGNING", 10.0 + 120.0).percent)  // window over, engine not yet NAVIGATING
        assertEquals(99, of("ALIGNING", 500.0).percent)
    }

    @Test
    fun aligningClampsBeforeInitAndWithoutInit() {
        assertEquals(0, of("ALIGNING", 5.0).percent)
        assertEquals(0, of("ALIGNING", 50.0, tInit = null).percent)
    }

    @Test
    fun navigatingIsLocked() {
        assertEquals(AlignmentStatus(AlignmentStatus.Phase.LOCKED, 100), of("NAVIGATING", 200.0))
    }

    @Test
    fun unknownModeIsUnavailable() {
        assertEquals(AlignmentStatus.UNAVAILABLE, of(null, 0.0))
        assertEquals(AlignmentStatus.UNAVAILABLE, of("SOMETHING", 0.0))
    }

    @Test(expected = IllegalArgumentException::class)
    fun nonPositiveWindowRaises() {
        AlignmentStatus.of("ALIGNING", 1.0, 0.0, windowS = 0.0)
    }
}
