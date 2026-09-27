package com.sih26168.deadreckoning.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.sih26168.deadreckoning.core.Display
import com.sih26168.deadreckoning.core.GnssState
import com.sih26168.deadreckoning.core.NavSnapshot

/**
 * Primary screen: track canvas + telemetry HUD + "Simulate GNSS outage" toggle.
 * The track canvas is a PLACEHOLDER for a map renderer: it draws the positions the engine
 * reported, in local metres, auto-scaled; there are no map tiles yet.
 */
@Composable
fun NavigationScreen(
    snapshot: NavSnapshot?,
    track: List<Pair<Double, Double>>,
    outageOn: Boolean,
    running: Boolean,
    onOutage: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        TrackCanvas(track, Modifier.fillMaxWidth().weight(1f))
        Hud(snapshot, outageOn)
        Card(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Simulate GNSS outage", fontWeight = FontWeight.Bold)
                    Text("Withholds every fix from the engine; still logged, flagged withheld_by_sim",
                        style = MaterialTheme.typography.bodySmall)
                }
                Switch(checked = outageOn, onCheckedChange = onOutage, enabled = running)
            }
        }
    }
}

@Composable
private fun Hud(s: NavSnapshot?, outageOn: Boolean) {
    val stateColor = when (s?.state) {
        GnssState.GOOD -> Color(0xFF2E7D32)
        GnssState.DEGRADED -> Color(0xFFF9A825)
        GnssState.RECOVERING -> Color(0xFF1565C0)
        GnssState.LOST -> Color(0xFFC62828)
        null -> Color.Gray
    }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(Display.speedKmh(s?.speedMps), style = MaterialTheme.typography.displaySmall,
                    modifier = Modifier.weight(1f))
                Text(Display.stateLabel(s?.state, outageOn), color = Color.White, fontWeight = FontWeight.Bold,
                    modifier = Modifier.background(stateColor).padding(horizontal = 8.dp, vertical = 4.dp))
            }
            HudRow("Confidence (within 10 m)", Display.percent(s?.confidence))
            HudRow("Horizontal σ", Display.metres(s?.sigmaHm))
            HudRow("Road", Display.roadName(s?.roadName))
            HudRow("Engine", s?.engineName ?: Display.NA)
            s?.note?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
        }
    }
}

@Composable
internal fun HudRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun TrackCanvas(track: List<Pair<Double, Double>>, modifier: Modifier) {
    Card(modifier) {
        Canvas(Modifier.fillMaxSize().padding(16.dp)) {
            if (track.isEmpty()) return@Canvas
            val xs = track.map { it.first }
            val ys = track.map { it.second }
            val span = maxOf(xs.max() - xs.min(), ys.max() - ys.min(), 50.0) // >= 50 m view
            val cx = (xs.max() + xs.min()) / 2
            val cy = (ys.max() + ys.min()) / 2
            val scale = (minOf(size.width, size.height) / span).toFloat()
            fun p(e: Double, n: Double) = Offset(
                size.width / 2 + ((e - cx) * scale).toFloat(),
                size.height / 2 - ((n - cy) * scale).toFloat(), // north up
            )
            val path = Path()
            track.forEachIndexed { i, (e, n) ->
                val o = p(e, n)
                if (i == 0) path.moveTo(o.x, o.y) else path.lineTo(o.x, o.y)
            }
            drawPath(path, Color(0xFF1565C0), style = Stroke(width = 4f))
            val (le, ln) = track.last()
            drawCircle(Color(0xFFC62828), radius = 10f, center = p(le, ln))
        }
    }
}
