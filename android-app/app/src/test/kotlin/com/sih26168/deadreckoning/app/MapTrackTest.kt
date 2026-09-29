package com.sih26168.deadreckoning.app

import com.sih26168.deadreckoning.core.GnssState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MapTrackTest {
    private fun p(i: Int, dr: Boolean) = GeoTrackPoint(i.toDouble(), 0.0, dr)

    @Test
    fun emptyTrackHasNoRuns() {
        assertEquals(emptyList<Pair<Boolean, List<GeoTrackPoint>>>(), trackRuns(emptyList()))
    }

    @Test
    fun singleModeIsOneRun() {
        val pts = listOf(p(0, false), p(1, false), p(2, false))
        assertEquals(listOf(false to pts), trackRuns(pts))
    }

    @Test
    fun runsShareTheBoundaryPoint() {
        val pts = listOf(p(0, false), p(1, false), p(2, true), p(3, true), p(4, false))
        assertEquals(
            listOf(
                false to listOf(p(0, false), p(1, false)),
                true to listOf(p(1, false), p(2, true), p(3, true)),
                false to listOf(p(3, true), p(4, false)),
            ),
            trackRuns(pts),
        )
    }

    @Test
    fun lostBeforeInitialisationIsNotDeadReckoning() {
        assertFalse(gnssOutShown(GnssState.LOST, MODE_WAITING, outageOn = false, running = true))
        assertTrue(gnssOutShown(GnssState.LOST, "NAVIGATING", outageOn = false, running = true))
        assertTrue(gnssOutShown(GnssState.LOST, null, outageOn = false, running = true)) // GNSS-only engine
    }

    @Test
    fun outageToggleOffFollowsTheEngine() {
        assertTrue(gnssOutShown(GnssState.GOOD, "NAVIGATING", outageOn = true, running = true))
        assertFalse(gnssOutShown(GnssState.GOOD, "NAVIGATING", outageOn = false, running = true))
        assertFalse(gnssOutShown(GnssState.RECOVERING, "NAVIGATING", outageOn = false, running = true))
    }

    @Test
    fun stoppedSessionShowsNothing() {
        assertFalse(gnssOutShown(GnssState.LOST, "NAVIGATING", outageOn = true, running = false))
    }
}
