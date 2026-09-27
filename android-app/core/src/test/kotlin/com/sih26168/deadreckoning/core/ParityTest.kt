package com.sih26168.deadreckoning.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

/**
 * Golden-vector parity with the Python reference (AGENTS.md 4). The vectors are generated
 * by scripts/parity/generate_golden_vectors.py from navcore and committed under
 * tests/regression/golden/; the Python side checks they are still what it produces.
 */
class ParityTest {
    private fun golden(name: String): JSONObject {
        val root = System.getProperty("sih26168.repoRoot") ?: File("../..").absolutePath
        val f = File(root, "tests/regression/golden/$name")
        assertTrue("golden vector missing: $f (run the Python generator)", f.isFile)
        return JSONObject(f.readText())
    }

    @Test
    fun `gnss state machine matches the python reference on every event`() {
        val g = golden("gnss_state_machine.json")
        val c = g.getJSONObject("config")
        val sm = GnssStateMachine(
            GnssStateConfig(
                degradedAccuracyM = c.getDouble("degraded_accuracy_m"), degradedAgeS = c.getDouble("degraded_age_s"),
                lostAgeS = c.getDouble("lost_age_s"), goodStreak = c.getInt("good_streak"),
                recoverStreak = c.getInt("recover_streak"),
            ),
        )
        assertEquals(GnssState.valueOf(g.getString("initial")), sm.state)
        val ev = g.getJSONArray("events")
        assertTrue(ev.length() > 400)
        for (i in 0 until ev.length()) {
            val e = ev.getJSONObject(i)
            val got = when (e.getString("op")) {
                "fix" -> sm.onFix(e.getDouble("t"), e.getDouble("accuracy_m"), e.getBoolean("accepted"))
                "time" -> sm.onTime(e.getDouble("t"))
                else -> error("unknown op")
            }
            assertEquals("event $i: $e", GnssState.valueOf(e.getString("expect")), got)
        }
    }

    @Test
    fun `gnss sample validation accepts and rejects exactly what the reference does`() {
        val cases = golden("sample_validation.json").getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val inp = c.getJSONObject("input")
            fun opt(k: String): Double? = if (inp.has(k)) inp.getDouble(k) else null
            val accepted = try {
                GnssSample(
                    tS = inp.getDouble("t_s"), latRad = inp.getDouble("lat_rad"), lonRad = inp.getDouble("lon_rad"),
                    hM = inp.getDouble("h_m"), horizontalAccuracyM = inp.getDouble("horizontal_accuracy_m"),
                    tReceivedS = opt("t_received_s") ?: inp.getDouble("t_s"),
                    speedMps = opt("speed_mps"), bearingRad = opt("bearing_rad"),
                )
                true
            } catch (x: SensorSampleException) {
                false
            }
            if (accepted != c.getBoolean("accepted")) fail("case ${c.getString("name")}: kotlin=$accepted python=${!accepted}")
        }
    }

    @Test
    fun `wgs84 radii match the python geodesy to 1e-12 relative`() {
        val rows = golden("wgs84_radii.json").getJSONArray("rows")
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            val lat = r.getDouble("lat_rad")
            val m = r.getDouble("meridian_m")
            val n = r.getDouble("prime_vertical_m")
            assertEquals("meridian @ $lat", m, TrackProjector.meridianRadius(lat), 1e-12 * m)
            assertEquals("prime vertical @ $lat", n, TrackProjector.primeVerticalRadius(lat), 1e-12 * n)
        }
    }

    @Test
    fun `filter policies equal the reference table`() {
        // navcore/state/gnss_state.py POLICY, value for value
        assertEquals(FilterPolicy(true, 1.0, false, true), POLICY[GnssState.GOOD])
        assertEquals(FilterPolicy(true, 2.0, true, true), POLICY[GnssState.DEGRADED])
        assertEquals(FilterPolicy(false, 1.0, true, true), POLICY[GnssState.LOST])
        assertEquals(FilterPolicy(true, 3.0, true, true), POLICY[GnssState.RECOVERING])
        assertEquals("DEAD_RECKONING", GnssState.LOST.displayName)
    }
}
