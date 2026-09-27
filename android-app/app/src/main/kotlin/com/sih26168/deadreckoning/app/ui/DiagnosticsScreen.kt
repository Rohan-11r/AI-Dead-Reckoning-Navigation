package com.sih26168.deadreckoning.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.sih26168.deadreckoning.app.DiagnosticsState
import com.sih26168.deadreckoning.core.Channel
import com.sih26168.deadreckoning.core.Display
import java.util.Locale

/** Developer diagnostics: measured rates, raw vectors, counters, EKF P diagonal, AI latency. */
@Composable
fun DiagnosticsScreen(d: DiagnosticsState, modifier: Modifier = Modifier) {
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        d.error?.let { Text(it, color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold) }
        if (d.engineNotes.isNotEmpty()) {
            Section("Engine resources") {
                for (n in d.engineNotes) Text(n, style = MaterialTheme.typography.bodySmall)
            }
        }
        Section("Sensor rates (measured on the sensors' own timestamps)") {
            for (c in Channel.entries) HudRow("${c.name} [${c.unit}]", Display.hz(d.ratesHz[c]))
            HudRow("GNSS fixes", Display.hz(d.gnssRateHz))
            HudRow("Available", d.sensorsAvailable.entries.joinToString { "${it.key}=${if (it.value) "yes" else "NO"}" })
        }
        Section("Raw readings (device frame)") {
            for (c in Channel.entries) HudRow(c.name, Display.vec(d.latest[c]))
        }
        Section("Counters (nothing is dropped silently)") {
            HudRow("sensor samples mapped", d.sensorMapped.toString())
            HudRow("sensor samples rejected", d.sensorRejected.toString())
            HudRow("sensor samples dropped (consumer stalled)", d.sensorDropped.toString())
            HudRow("IMU samples without a fresh gyro", d.imuWithoutGyro.toString())
            HudRow("10 Hz bins skipped (a channel missing)", d.resamplerSkipped.toString())
            HudRow("GNSS fixes mapped / rejected", "${d.fixesAccepted} / ${d.fixesRejected}")
            HudRow("GNSS fixes withheld by simulated outage", d.fixesWithheldBySim.toString())
            d.lastSensorRejection?.let { HudRow("last sensor rejection", it) }
            d.lastFixRejection?.let { HudRow("last fix rejection", it) }
        }
        Section("EKF covariance diagonal") {
            val p = d.covarianceDiag
            if (p == null) {
                Text("n/a: no dead-reckoning engine running (see Engine resources)",
                    style = MaterialTheme.typography.bodySmall)
            } else {
                val labels = listOf("pE", "pN", "pU", "vE", "vN", "vU", "ψE", "ψN", "ψU", "baX", "baY", "baZ", "bgX", "bgY", "bgZ")
                p.forEachIndexed { i, v -> HudRow(labels.getOrElse(i) { "x$i" }, String.format(Locale.ROOT, "%.3e", v)) }
            }
        }
        Section("AI inference latency") {
            HudRow("live (last Model A inference in the engine)", Display.ms(d.liveAiLatencyMs))
            val a = d.aiLatency
            if (a == null) {
                Text("measuring…", style = MaterialTheme.typography.bodySmall)
            } else {
                HudRow("model", a.model)
                HudRow("median / p90", "${Display.ms(a.medianMs)} / ${Display.ms(a.p90Ms)}")
                HudRow("runs", a.runs.toString())
                Text(a.note, style = MaterialTheme.typography.bodySmall)
            }
        }
        Section("Session log") {
            HudRow("sensor lines", d.loggerSensorLines.toString())
            HudRow("GNSS lines", d.loggerGnssLines.toString())
            Text(d.sessionDir ?: Display.NA, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, fontWeight = FontWeight.Bold)
            content()
        }
    }
}
