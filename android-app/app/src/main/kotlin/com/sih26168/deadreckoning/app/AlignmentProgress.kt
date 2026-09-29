package com.sih26168.deadreckoning.app

/**
 * How far the engine's OWN heading alignment has got -- read from the engine, never computed
 * beside it. No new math: the mount tilt (gravity) and the heading (8 hypotheses scored
 * against GNSS for `EngineConfig.mhWindowS`) are resolved inside DrEngine, unchanged.
 *
 *   WAITING    no GNSS fix > initMinSpeedMps with a bearing yet             -> 0 %
 *   ALIGNING   (tS - tInit) / mhWindowS                                      -> 0..99 %
 *   NAVIGATING one heading hypothesis kept                                   -> 100 %
 *
 * Why no separate calibrator: Phase 9 measured a whole-drive (look-ahead) mount against the
 * shipped causal one and found no consistent gain (reports/phase9/outage_benchmark.json,
 * arm given_mount); and pre-rotating the IMU would change the ONNX speed model's
 * device-frame inputs away from what it was trained on.
 */
data class AlignmentStatus(val phase: Phase, val percent: Int) {
    enum class Phase { UNAVAILABLE, WAITING_FOR_GNSS, ALIGNING, LOCKED }

    companion object {
        val UNAVAILABLE = AlignmentStatus(Phase.UNAVAILABLE, 0)

        /** [mode] and [tInitS] from DrEngine; [tS] = the snapshot time on the same clock. */
        fun of(mode: String?, tS: Double, tInitS: Double?, windowS: Double): AlignmentStatus = when (mode) {
            "NAVIGATING" -> AlignmentStatus(Phase.LOCKED, 100)
            "ALIGNING" -> {
                require(windowS > 0.0) { "alignment window must be > 0 s, got $windowS" }
                val frac = if (tInitS == null || !tS.isFinite()) 0.0 else ((tS - tInitS) / windowS).coerceIn(0.0, 1.0)
                // 100 % is reserved for the engine's own switch to NAVIGATING
                AlignmentStatus(Phase.ALIGNING, (frac * 100.0).toInt().coerceAtMost(99))
            }
            "WAITING" -> AlignmentStatus(Phase.WAITING_FOR_GNSS, 0)
            else -> UNAVAILABLE
        }
    }
}
