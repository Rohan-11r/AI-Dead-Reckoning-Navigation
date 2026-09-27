package com.sih26168.deadreckoning.core.nav

import com.sih26168.deadreckoning.core.GnssSample
import com.sih26168.deadreckoning.core.GnssState
import com.sih26168.deadreckoning.core.mapmatch.DeadReckoningMapMatcher
import com.sih26168.deadreckoning.core.mapmatch.HmmMapMatcher
import com.sih26168.deadreckoning.core.mapmatch.RoadNetwork
import com.sih26168.deadreckoning.core.mapmatch.courseAndSigma
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Phase 11 golden-vector parity: the Kotlin port against the Python reference, layer by layer
 * (AGENTS.md 4). Vectors: tests/regression/golden/nav_*.json (scripts/parity/nav_golden.py).
 * Tolerances are per layer: tight for single operations, looser for the 1,800-step pipeline
 * (different summation order accumulates); discrete outcomes (GNSS state, gate decisions,
 * counters, segment ids, modes) must match EXACTLY.
 */
class NavParityTest {
    private val root = File(System.getProperty("sih26168.repoRoot") ?: File("../..").absolutePath)
    private fun golden(name: String) = JSONObject(File(root, "tests/regression/golden/$name").readText())

    private fun arr(a: JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }
    private fun close(msg: String, want: Double, got: Double, atol: Double, rtol: Double = 0.0) =
        assertTrue("$msg: want $want got $got", abs(want - got) <= atol + rtol * abs(want))
    private fun close(msg: String, want: DoubleArray, got: DoubleArray, atol: Double, rtol: Double = 0.0) {
        assertEquals("$msg length", want.size, got.size)
        for (i in want.indices) close("$msg[$i]", want[i], got[i], atol, rtol)
    }

    private fun noise(o: JSONObject) = NoiseParams(o.getDouble("accel_white_mps2_rtHz"), o.getDouble("gyro_white_radps_rtHz"),
        o.getDouble("accel_bias_sigma_mps2"), o.getDouble("accel_bias_tau_s"), o.getDouble("gyro_bias_sigma_radps"),
        o.getDouble("gyro_bias_tau_s"), o.getDouble("unmeasured_rate_radps_rtHz"))

    private fun state(o: JSONObject) = NavState(o.getDouble("t"), o.getDouble("lat"), o.getDouble("lon"), o.getDouble("h"),
        arr(o.getJSONArray("v")), arr(o.getJSONArray("q")))

    private fun optD(o: JSONObject, k: String): Double? = if (o.isNull(k)) null else o.getDouble(k)

    private fun fix(o: JSONObject) = GnssSample(o.getDouble("t"), o.getDouble("lat"), o.getDouble("lon"), o.getDouble("h"),
        o.getDouble("acc"), o.getDouble("t_rx"), optD(o, "speed"), optD(o, "bearing"))

    /** Position difference in metres (lat/lon radians -> m, generously with R = 6.4e6). */
    private fun assertState(msg: String, want: JSONObject, got: NavState, posM: Double, vel: Double, q: Double) {
        close("$msg lat[m]", want.getDouble("lat") * 6.4e6, got.latRad * 6.4e6, posM)
        close("$msg lon[m]", want.getDouble("lon") * 6.4e6, got.lonRad * 6.4e6, posM)
        close("$msg h", want.getDouble("h"), got.hM, posM)
        close("$msg v", arr(want.getJSONArray("v")), got.vEnu, vel)
        // q and -q are the same rotation; the reference keeps the propagated sign
        close("$msg q", arr(want.getJSONArray("q")), got.qNb, q)
    }

    @Test
    fun `quaternions and geodesy match the reference`() {
        val g = golden("nav_geometry.json")
        val qs = g.getJSONArray("quaternion")
        for (i in 0 until qs.length()) {
            val c = qs.getJSONObject(i)
            val p = arr(c.getJSONArray("p"))
            val q = arr(c.getJSONArray("q"))
            close("mul $i", arr(c.getJSONArray("mul")), Quat.mul(p, q), 1e-15)
            close("rotate $i", arr(c.getJSONArray("rotate")), Quat.rotate(q, arr(c.getJSONArray("v"))), 1e-13)
            close("dcm $i", arr(c.getJSONArray("dcm")), Quat.toDcm(q).a, 1e-15)
            close("dcm_to_q $i", arr(c.getJSONArray("dcm_to_q")), Quat.fromDcm(Quat.toDcm(q)), 1e-14)
            close("from_rotvec $i", arr(c.getJSONArray("from_rotvec")), Quat.fromRotvec(arr(c.getJSONArray("rv"))), 1e-15)
            close("slerp $i", arr(c.getJSONArray("slerp")), Quat.slerp(p, q, c.getDouble("t")), 1e-14)
            val e = arr(c.getJSONArray("euler"))
            close("from_euler $i", arr(c.getJSONArray("from_euler")), Quat.fromEuler(e[0], e[1], e[2]), 1e-15)
        }
        val geo = g.getJSONArray("geodesy")
        for (i in 0 until geo.length()) {
            val c = geo.getJSONObject(i)
            val lat = c.getDouble("lat")
            val lon = c.getDouble("lon")
            val h = c.getDouble("h")
            close("ecef $i", arr(c.getJSONArray("ecef")), Geodesy.geodeticToEcef(lat, lon, h), 1e-8)
            val back = Geodesy.ecefToGeodetic(arr(c.getJSONArray("ecef")))
            close("back lat $i", c.getJSONArray("back").getDouble(0), back[0], 1e-13)
            close("back lon $i", c.getJSONArray("back").getDouble(1), back[1], 1e-13)
            close("back h $i", c.getJSONArray("back").getDouble(2), back[2], 1e-7)
            close("gravity $i", c.getDouble("gravity"), Geodesy.normalGravity(lat, h), 1e-14)
            close("earth rate $i", arr(c.getJSONArray("earth_rate")), Geodesy.earthRateEnu(lat), 1e-18)
            close("transport $i", arr(c.getJSONArray("transport_rate")), Geodesy.transportRateEnu(lat, h, doubleArrayOf(12.0, -7.0, 0.3)), 1e-18, 1e-12)
        }
        val l = g.getJSONObject("ltp")
        val ltp = LocalTangentPlane(l.getDouble("lat0"), l.getDouble("lon0"), l.getDouble("h0"))
        val pts = l.getJSONArray("points")
        for (i in 0 until pts.length()) {
            val p = pts.getJSONObject(i)
            val geoP = ltp.enuToGeodetic(arr(p.getJSONArray("enu")))
            close("ltp geo $i", arr(p.getJSONArray("geo")).copyOf(2), geoP.copyOf(2), 1e-13)
            close("ltp back $i", arr(p.getJSONArray("back")), ltp.geodeticToEnu(geoP[0], geoP[1], geoP[2]), 1e-6)
        }
    }

    @Test
    fun `ins propagation matches the reference over 300 steps`() {
        val g = golden("nav_ins.json")
        var s = state(g.getJSONObject("initial"))
        val steps = g.getJSONArray("steps")
        for (k in 0 until steps.length()) {
            val st = steps.getJSONObject(k)
            s = Ins.propagate(s, arr(st.getJSONArray("f")), arr(st.getJSONArray("w")), st.getDouble("dt"))
            assertState("step $k", st.getJSONObject("out"), s, posM = 1e-6, vel = 1e-9, q = 1e-11)
        }
    }

    @Test
    fun `ekf operations match the reference, including gate decisions`() {
        val g = golden("nav_ekf_ops.json")
        val ekf = ErrorStateEkf(state(g.getJSONObject("initial")), Mat.diag(arr(g.getJSONArray("P0_diag"))), noise(g.getJSONObject("noise")))
        val ops = g.getJSONArray("ops")
        for (k in 0 until ops.length()) {
            val o = ops.getJSONObject(k)
            val ok: Boolean? = when (o.getString("op")) {
                "predict" -> {
                    val va = o.getJSONArray("valid")
                    ekf.predict(FilterImu(o.getDouble("t"), arr(o.getJSONArray("acc")), arr(o.getJSONArray("gyro")),
                        BooleanArray(3) { va.getBoolean(it) }), o.getDouble("dt"))
                    null
                }
                "gnss" -> ekf.updateGnss(fix(o.getJSONObject("fix")))
                "zupt" -> ekf.updateZupt(o.getDouble("t"))
                "level" -> ekf.updateLevelling(o.getDouble("t"), arr(o.getJSONArray("acc")))
                "nhc" -> Nhc.apply(ekf, o.getDouble("t"), Mat(3, 3, arr(o.getJSONArray("R_vb"))), o.getBoolean("lateral"),
                    o.getDouble("lat_acc"), NhcConfig()) == "accepted"
                "update" -> ekf.update(o.getString("kind"), o.getDouble("t"), arr(o.getJSONArray("nu")),
                    Mat(1, N_STATES, arr(o.getJSONArray("H"))), Mat(1, 1, arr(o.getJSONArray("R"))))
                "reset" -> { resetToFix(ekf, fix(o.getJSONObject("fix"))); true }
                else -> error("op")
            }
            if (!o.isNull("ok")) assertEquals("op $k ${o.getString("op")} accepted", o.getBoolean("ok"), ok)
            if (!o.isNull("nis")) close("op $k nis", o.getDouble("nis"), ekf.log.last().nis, 1e-9, 1e-7)
            assertState("op $k", o.getJSONObject("state"), ekf.nominal, posM = 1e-6, vel = 1e-8, q = 1e-10)
            close("op $k P", arr(o.getJSONArray("P")), ekf.P.a, 1e-14, 1e-7)
            close("op $k b_a", arr(o.getJSONArray("b_a")), ekf.ba, 1e-12)
            close("op $k b_g", arr(o.getJSONArray("b_g")), ekf.bg, 1e-14)
        }
    }

    private fun window(c: JSONObject, w: Int): Triple<Array<DoubleArray>, Array<DoubleArray>, DoubleArray> {
        val a = arr(c.getJSONArray("acc"))
        val g = arr(c.getJSONArray("grav"))
        return Triple(Array(w) { doubleArrayOf(a[3 * it], a[3 * it + 1], a[3 * it + 2]) },
            Array(w) { doubleArrayOf(g[3 * it], g[3 * it + 1], g[3 * it + 2]) }, arr(c.getJSONArray("wz")))
    }

    @Test
    fun `features match the reference bit for bit in float32`() {
        val g = golden("nav_features.json")
        val cases = g.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val (acc, grav, wz) = window(c, 50)
            val gyro = Array(50) { doubleArrayOf(0.0, 0.0, wz[it]) }
            val got = Features.make(acc, grav, gyro, g.getDouble("accel_scale"), g.getDouble("gyro_scale"))
            val want = arr(c.getJSONArray("features"))
            for (k in want.indices) close("case $i feature $k", want[k], got[k].toDouble(), 1e-7, 2.0 * Math.ulp(1.0f).toDouble())
        }
    }

    @Test
    fun `model A through onnxruntime matches python within the phase 6 gate`() {
        val g = golden("nav_model_a.json")
        val dir = File(root, "models/exported")
        val name = g.getString("model")
        OnnxSpeedModel(File(dir, "$name.onnx").readBytes(), File(dir, "$name.model_card.json").readText()).use { m ->
            val cases = g.getJSONArray("cases")
            for (i in 0 until cases.length()) {
                val c = cases.getJSONObject(i)
                val (acc, grav, wz) = window(c, m.window)
                val gyro = Array(m.window) { doubleArrayOf(0.0, 0.0, wz[it]) }
                val (mean, logvar) = m.run(Features.make(acc, grav, gyro, m.accelScale, m.gyroScale))
                close("mean $i", c.getDouble("mean"), mean, 1e-5, 1e-6)
                close("logvar $i", c.getDouble("logvar"), logvar, 1e-5, 1e-6)
                val (pm, pv) = m.predict(acc, grav, wz)
                close("predict mean $i", c.getDouble("predict_mean"), pm, 1e-5, 1e-6)
                close("predict var $i", c.getDouble("predict_var"), pv, 1e-6, 1e-5)
            }
        }
    }

    /** The deterministic stand-in the reference used: 12 + 0.5 mean(acc_x) + 10 mean(wz), var 0.36. */
    private object StubSpeed : SpeedModel {
        override val window = 20
        override fun predict(acc: Array<DoubleArray>, grav: Array<DoubleArray>, wz: DoubleArray): Pair<Double, Double> =
            (12.0 + 0.5 * (acc.sumOf { it[0] } / acc.size) + 10.0 * (wz.sum() / wz.size)) to 0.36
    }

    private fun fixes(o: JSONObject): List<GnssSample> {
        val a = o.getJSONArray("fixes")
        return List(a.length()) { fix(a.getJSONObject(it)) }
    }

    @Test
    fun `fused navigator matches the reference through a gnss dropout`() {
        val g = golden("nav_navigator.json")
        val ekf = ErrorStateEkf(state(g.getJSONObject("initial")), Mat.diag(arr(g.getJSONArray("P0_diag"))), noise(g.getJSONObject("noise")))
        val nav = FusedNavigator(ekf, AxisMap.IOVNBD, FusionConfig(), StubSpeed, Rvb = Mat.eye(3), lateralNhcAllowed = true)
        val inputs = g.getJSONArray("inputs")
        val outs = g.getJSONArray("outputs")
        var oi = 0
        for (k in 0 until inputs.length()) {
            val st = inputs.getJSONObject(k)
            nav.step(st.getDouble("t"), arr(st.getJSONArray("acc")), arr(st.getJSONArray("grav")), arr(st.getJSONArray("gyro")), fixes(st))
            if (oi < outs.length() && outs.getJSONObject(oi).getInt("k") == k) {
                val o = outs.getJSONObject(oi++)
                assertEquals("step $k gnss state", GnssState.valueOf(o.getString("gnss_state")), nav.sm.state)
                assertState("step $k", o.getJSONObject("state"), nav.ekf.nominal, posM = 1e-3, vel = 1e-5, q = 1e-7)
                close("step $k P diag", arr(o.getJSONArray("P_diag")), nav.ekf.P.diag(), 1e-12, 1e-5)
            }
        }
        assertEquals(outs.length(), oi)
        val c = g.getJSONObject("counts")
        for (key in c.keys()) assertEquals("count $key", c.getLong(key), nav.counts[key] ?: 0L)
        assertEquals(c.length(), nav.counts.size)
    }

    @Test
    fun `streaming engine matches the reference, heading hypotheses included`() {
        val g = golden("nav_engine.json")
        val eng = DrEngine(noise(g.getJSONObject("noise")), EngineConfig(mhWindowS = 40.0), StubSpeed, AxisMap.IOVNBD)
        val inputs = g.getJSONArray("inputs")
        val outs = g.getJSONArray("outputs")
        var oi = 0
        for (k in 0 until inputs.length()) {
            val st = inputs.getJSONObject(k)
            val o = eng.onSample(st.getDouble("t"), arr(st.getJSONArray("acc")), arr(st.getJSONArray("grav")), arr(st.getJSONArray("gyro")), fixes(st))
            if (oi < outs.length() && outs.getJSONObject(oi).getInt("k") == k) {
                val w = outs.getJSONObject(oi++)
                assertTrue("output at $k", o != null)
                assertEquals("mode $k", w.getString("mode"), o!!.mode)
                assertEquals("gnss $k", GnssState.valueOf(w.getString("gnss_state")), o.gnssState)
                close("lat $k [m]", w.getDouble("lat") * 6.4e6, o.latRad * 6.4e6, 1e-3)
                close("lon $k [m]", w.getDouble("lon") * 6.4e6, o.lonRad * 6.4e6, 1e-3)
                close("sigma $k", w.getDouble("sigma_h"), o.sigmaHm, 1e-9, 1e-5)
                close("v $k", arr(w.getJSONArray("v")), o.vEnu, 1e-5)
            }
        }
        assertEquals(outs.length(), oi)
        assertEquals(g.getDouble("chosen_offset_rad"), eng.chosenOffsetRad!!, 0.0)
        assertEquals(g.getString("tilt_source"), eng.tiltSource)
        assertEquals(g.getDouble("t_init"), eng.tInit!!, 0.0)
    }

    @Test
    fun `map matcher matches the reference on every epoch`() {
        val g = golden("nav_map_matching.json")
        val gold = File(root, "tests/regression/golden")
        val cases = g.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val net = File(gold, c.getString("network")).inputStream().use { RoadNetwork.read(it) }
            val m = HmmMapMatcher(net)
            val sig2 = c.getDouble("sigma").let { it * it }
            val obs = c.getJSONArray("obs")
            val res = c.getJSONArray("results")
            for (k in 0 until obs.length()) {
                val ob = obs.getJSONArray(k)
                val h = if (ob.isNull(2)) null else ob.getDouble(2)
                val r = m.step(k.toDouble(), ob.getDouble(0), ob.getDouble(1), doubleArrayOf(sig2, 0.0, 0.0, sig2), h,
                    if (h != null) c.getDouble("speed") else 0.0)
                val w = res.getJSONObject(k)
                val tag = "${c.getString("label")} epoch $k"
                assertEquals("$tag mode", w.getString("mode"), r.mode)
                assertEquals("$tag matched", w.getBoolean("matched"), r.matched)
                assertEquals("$tag n candidates", w.getInt("n_candidates"), r.nCandidates)
                close("$tag radius", w.getDouble("radius"), r.searchRadiusM, 1e-9)
                if (r.matched) {
                    assertEquals("$tag segment", w.getInt("segment"), r.segment)
                    close("$tag x", w.getDouble("x"), r.xM!!, 1e-9)
                    close("$tag y", w.getDouble("y"), r.yM!!, 1e-9)
                    close("$tag lat", w.getDouble("lat"), r.latRad!!, 1e-13)
                    close("$tag lon", w.getDouble("lon"), r.lonRad!!, 1e-13)
                    close("$tag confidence", w.getDouble("confidence"), r.confidence, 1e-9)
                }
            }
            val vit = c.getJSONArray("viterbi")
            val path = m.decode()
            assertEquals("${c.getString("label")} viterbi length", vit.length(), path.size)
            for (k in 0 until vit.length()) assertEquals("viterbi $k", vit.getJSONObject(k).getInt("segment"), path[k].second.seg)
            val counts = c.getJSONObject("counts")
            for (key in counts.keys()) assertEquals("${c.getString("label")} count $key", counts.getInt(key), m.counts[key] ?: 0)
        }
        val dr = g.getJSONObject("dr_matcher")
        val net = File(gold, dr.getString("network")).inputStream().use { RoadNetwork.read(it) }
        val mm = DeadReckoningMapMatcher(net)
        val feed = dr.getJSONArray("feed")
        for (k in 0 until feed.length()) {
            val f = feed.getJSONObject(k)
            val r = mm.update(f.getDouble("t"), f.getDouble("lat"), f.getDouble("lon"), arr(f.getJSONArray("P")),
                arr(f.getJSONArray("v")), arr(f.getJSONArray("Pv")))
            if (f.isNull("out")) {
                assertTrue("dr $k rate-limited", r == null)
            } else {
                val w = f.getJSONObject("out")
                assertEquals("dr $k matched", w.getBoolean("matched"), r!!.matched)
                assertEquals("dr $k segment", w.getInt("segment"), r.segment)
                close("dr $k lat", w.getDouble("lat"), r.latRad!!, 1e-13)
                close("dr $k confidence", w.getDouble("confidence"), r.confidence, 1e-9)
            }
        }
        val course = g.getJSONArray("course")
        for (i in 0 until course.length()) {
            val c = course.getJSONObject(i)
            close("course $i", arr(c.getJSONArray("out")), courseAndSigma(arr(c.getJSONArray("v")), arr(c.getJSONArray("Pv"))), 1e-15)
        }
    }

    @Test
    fun `the device path (channels, 10 Hz resampler, fix queue, engine) reproduces the reference`() {
        // Every golden 10 Hz sample is fed as three channel events in the MIDDLE of its bin, so the
        // resampler's bin mean is the sample itself; fixes are queued as they arrive.
        val g = golden("nav_engine.json")
        val nav = DeadReckoningNavigation(noise(g.getJSONObject("noise")), EngineConfig(mhWindowS = 40.0),
            axisMap = AxisMap.IOVNBD, testModel = StubSpeed)
        val byStep = HashMap<Long, EngineOutput>()
        nav.onOutput = { o -> byStep[Math.round(o.tS * 10)] = o }
        val inputs = g.getJSONArray("inputs")
        val ch = com.sih26168.deadreckoning.core.Channel
        fun vec(a: JSONArray) = com.sih26168.deadreckoning.core.Vec3(a.getDouble(0), a.getDouble(1), a.getDouble(2))
        for (k in 0 until inputs.length()) {
            val st = inputs.getJSONObject(k)
            fixes(st).forEach { nav.onFix(it) }
            val t = st.getDouble("t") - 0.05
            nav.onChannel(com.sih26168.deadreckoning.core.ChannelSample(t, ch.ACCEL, vec(st.getJSONArray("acc"))))
            nav.onChannel(com.sih26168.deadreckoning.core.ChannelSample(t, ch.GRAVITY, vec(st.getJSONArray("grav"))))
            nav.onChannel(com.sih26168.deadreckoning.core.ChannelSample(t, ch.GYRO, vec(st.getJSONArray("gyro"))))
        }
        val outs = g.getJSONArray("outputs")
        var compared = 0
        for (i in 0 until outs.length()) {
            val w = outs.getJSONObject(i)
            val o = byStep[w.getInt("k").toLong()] ?: continue  // the last sample's bin never closes
            assertEquals("mode ${w.getInt("k")}", w.getString("mode"), o.mode)
            assertEquals("gnss ${w.getInt("k")}", GnssState.valueOf(w.getString("gnss_state")), o.gnssState)
            close("lat ${w.getInt("k")} [m]", w.getDouble("lat") * 6.4e6, o.latRad * 6.4e6, 1e-3)
            close("lon ${w.getInt("k")} [m]", w.getDouble("lon") * 6.4e6, o.lonRad * 6.4e6, 1e-3)
            compared++
        }
        assertTrue("compared $compared of ${outs.length()}", compared >= outs.length() - 1)
        assertEquals(0L, nav.resampler.nSkippedIncomplete)
        val snap = nav.snapshot()
        assertTrue(snap.latRad != null && snap.covarianceDiag != null && snap.mode == "NAVIGATING")
    }

    @Test
    fun `ten hz resampler averages bins and never fills a missing channel`() {
        val r = TenHzResampler()
        val ch = com.sih26168.deadreckoning.core.Channel
        val out = mutableListOf<TenHzResampler.Sample>()
        fun feed(t: Double, c: com.sih26168.deadreckoning.core.Channel, v: Double) {
            out += r.offer(com.sih26168.deadreckoning.core.ChannelSample(t, c, com.sih26168.deadreckoning.core.Vec3(v, 0.0, 9.8)))
        }
        feed(0.01, ch.ACCEL, 1.0); feed(0.05, ch.ACCEL, 3.0); feed(0.02, ch.GYRO, 0.1); feed(0.03, ch.GRAVITY, 0.0)
        feed(0.12, ch.ACCEL, 5.0)  // closes bin (0, 0.1]
        assertEquals(1, out.size)
        assertEquals(0.1, out[0].tS, 1e-12)
        assertEquals(2.0, out[0].acc[0], 1e-12)  // mean of 1 and 3
        feed(0.25, ch.ACCEL, 7.0)  // bin (0.1, 0.2] had no gyro/gravity: skipped, not filled
        assertEquals(1, out.size)
        assertEquals(1L, r.nSkippedIncomplete)
        assertEquals(1L, r.nEmitted)
    }
}
