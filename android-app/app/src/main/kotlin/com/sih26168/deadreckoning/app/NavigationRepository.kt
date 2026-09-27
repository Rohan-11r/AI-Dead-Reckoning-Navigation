package com.sih26168.deadreckoning.app

import android.app.Application
import com.sih26168.deadreckoning.core.Channel
import com.sih26168.deadreckoning.core.NavSnapshot
import com.sih26168.deadreckoning.core.Vec3
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Measured model inference time (see OnnxLatencyProbe for exactly what is measured). */
data class AiLatency(
    val model: String,
    val medianMs: Double?,
    val p90Ms: Double?,
    val runs: Int,
    val note: String,
)

/** Everything the diagnostics screen shows; published by the acquisition service. */
data class DiagnosticsState(
    val ratesHz: Map<Channel, Double> = emptyMap(),
    val latest: Map<Channel, Vec3> = emptyMap(),
    val gnssRateHz: Double = 0.0,
    val sensorsAvailable: Map<String, Boolean> = emptyMap(),
    val sensorMapped: Long = 0,
    val sensorRejected: Long = 0,
    val sensorDropped: Long = 0,
    val lastSensorRejection: String? = null,
    val imuWithoutGyro: Long = 0,
    val fixesAccepted: Long = 0,
    val fixesRejected: Long = 0,
    val lastFixRejection: String? = null,
    val fixesWithheldBySim: Long = 0,
    val loggerSensorLines: Long = 0,
    val loggerGnssLines: Long = 0,
    val sessionDir: String? = null,
    val covarianceDiag: DoubleArray? = null,
    val aiLatency: AiLatency? = null,
    val error: String? = null,
)

/**
 * Process-wide state between the acquisition service (writer, its own coroutines) and the UI
 * (reader). StateFlows conflate: the UI sees the latest value at its own frame rate, however
 * fast sensors arrive -- the decoupling the design asks for.
 */
class NavigationRepository {
    private val _snapshot = MutableStateFlow<NavSnapshot?>(null)
    val snapshot: StateFlow<NavSnapshot?> = _snapshot.asStateFlow()

    private val _diagnostics = MutableStateFlow(DiagnosticsState())
    val diagnostics: StateFlow<DiagnosticsState> = _diagnostics.asStateFlow()

    private val _track = MutableStateFlow<List<Pair<Double, Double>>>(emptyList())
    val track: StateFlow<List<Pair<Double, Double>>> = _track.asStateFlow()

    private val _running = MutableStateFlow(false)
    val running: StateFlow<Boolean> = _running.asStateFlow()

    /** The "Simulate GNSS outage" toggle, as requested by the UI. */
    val outageRequested = MutableStateFlow(false)

    fun publish(snapshot: NavSnapshot?, diagnostics: DiagnosticsState, track: List<Pair<Double, Double>>) {
        _snapshot.value = snapshot
        _diagnostics.value = diagnostics
        _track.value = track
    }

    fun setRunning(on: Boolean) {
        _running.value = on
    }

    fun reportError(message: String) {
        _diagnostics.value = _diagnostics.value.copy(error = message)
    }
}

class NavApplication : Application() {
    val repository = NavigationRepository()
}
