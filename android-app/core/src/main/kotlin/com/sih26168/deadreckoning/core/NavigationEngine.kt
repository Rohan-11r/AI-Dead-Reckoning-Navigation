package com.sih26168.deadreckoning.core

/**
 * What the app shows about navigation, from whichever engine is running.
 * Null means "not available", never a placeholder number.
 */
data class NavSnapshot(
    val tS: Double,
    val state: GnssState,
    val latRad: Double?,
    val lonRad: Double?,
    val speedMps: Double?,
    val sigmaHm: Double?,
    val confidence: Double?,            // positionConfidence(sigmaHm)
    val roadName: String?,              // map matcher: not on device yet -> null
    val covarianceDiag: DoubleArray?,   // 15-state EKF P diagonal: not on device yet -> null
    val aiLatencyMs: Double?,           // measured model inference time, when the model runs
    val engineName: String,
    val note: String? = null,
)

/** The on-device navigation engine contract. The streaming Python reference is
 * `navcore.fusion.engine.NavigationEngine`; its port is the edge-engine phase. */
interface NavigationEngine {
    val name: String
    fun onImu(sample: ImuSample)
    fun onChannel(sample: ChannelSample) {}
    fun onFix(fix: GnssSample)
    fun snapshot(): NavSnapshot
}

/**
 * HONEST PLACEHOLDER engine for Phase 10 (base & sensors). It runs the ported GNSS state
 * machine and reports the latest GNSS fix -- it does NOT dead-reckon: the INS/EKF/AI stack
 * is not ported to the device yet (golden-vector parity first, AGENTS.md 4). So in LOST
 * ("DEAD_RECKONING") it reports NO position, rather than a stale or invented one, and says
 * why. With no filter there is no NIS gate: every fix counts as accepted.
 */
class GnssOnlyEngine(cfg: GnssStateConfig = GnssStateConfig()) : NavigationEngine {
    override val name = "gnss-only (DR engine not yet on device)"
    val sm = GnssStateMachine(cfg)
    private var lastFix: GnssSample? = null
    private var lastT = 0.0

    override fun onImu(sample: ImuSample) {
        lastT = sample.tS
        sm.onTime(sample.tS)
    }

    override fun onFix(fix: GnssSample) {
        lastT = maxOf(lastT, fix.tReceivedS)
        sm.onFix(fix.tReceivedS, fix.horizontalAccuracyM, accepted = true)
        lastFix = fix
    }

    override fun snapshot(): NavSnapshot {
        val f = lastFix
        val lost = sm.state == GnssState.LOST
        val sigma = if (f == null || lost) null else sigmaFromAndroidAccuracy(f.horizontalAccuracyM)
        return NavSnapshot(
            tS = lastT, state = sm.state,
            latRad = if (lost) null else f?.latRad, lonRad = if (lost) null else f?.lonRad,
            speedMps = if (lost) null else f?.speedMps,
            sigmaHm = sigma, confidence = positionConfidence(sigma),
            roadName = null, covarianceDiag = null, aiLatencyMs = null, engineName = name,
            note = if (lost) "GNSS lost: the dead-reckoning engine is not on the device yet" else null,
        )
    }
}
