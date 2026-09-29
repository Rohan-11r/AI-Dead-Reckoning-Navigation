package com.sih26168.deadreckoning.app.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sih26168.deadreckoning.app.NhcStatus
import com.sih26168.deadreckoning.app.VibrationStatus
import com.sih26168.deadreckoning.app.ui.theme.RouteonOrange
import com.sih26168.deadreckoning.core.Display
import java.util.Locale

/**
 * "Real-Time AI Diagnostics": a dark glass card over the map with three live indicators, each
 * a readout of something the engine really does -- nothing here is simulated:
 *  - Vibration & shock filter: raw-accelerometer spikes found by [com.sih26168.deadreckoning.app.ShockMonitor]
 *    and what is left of each after the engine's 100 ms averaging (the INS input), in m/s^2.
 *  - AI speed model: running when the engine reports a measured Model A inference time.
 *  - NHC + map matching: the engine's NHC update outcome (its own counters) and the HMM map
 *    matcher's confidence. NHC constrains the VELOCITY (no vertical motion in the vehicle
 *    frame); the road snap is the map matcher's, shown as the green track.
 * Tap the header to collapse; a shock warning still shows while collapsed.
 */
@Composable
internal fun AiDiagnosticsCard(
    vibration: VibrationStatus,
    nhc: NhcStatus,
    aiLatencyMs: Double?,
    hasMap: Boolean,
    mapMatchConfidence: Double?,
    roadName: String?,
    deadReckoning: Boolean,
    modifier: Modifier = Modifier,
) {
    var expanded by rememberSaveable { mutableStateOf(true) }
    val shape = RoundedCornerShape(16.dp)
    val flashing = vibration.flashing()
    Column(
        modifier
            .fillMaxWidth()
            // glass: translucent dark gradient + a hairline light edge
            .background(Brush.verticalGradient(listOf(GLASS_TOP, GLASS_BOTTOM)), shape)
            .border(1.dp, Color.White.copy(alpha = 0.12f), shape)
            .padding(horizontal = 14.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .clickable(onClickLabel = if (expanded) "Collapse diagnostics" else "Expand diagnostics") { expanded = !expanded },
            verticalAlignment = Alignment.CenterVertically,
        ) {
            LiveDot(if (vibration.active) OK else IDLE)
            Spacer(Modifier.width(8.dp))
            Text("REAL-TIME AI DIAGNOSTICS", color = Color.White, fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.labelMedium, letterSpacing = 1.2.sp, modifier = Modifier.weight(1f))
            Text(if (expanded) "▾" else "▸", color = MUTED, style = MaterialTheme.typography.labelMedium)
        }
        AnimatedVisibility(visible = expanded) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                IndicatorRow(
                    title = if (vibration.active) "Vibration & Shock Filter: Active" else "Vibration & Shock Filter: Off",
                    detail = if (vibration.active) {
                        "Raw accelerometer → 100 ms averaging → INS  ·  ${vibration.nShocks} spikes detected"
                    } else "No dead-reckoning engine running",
                    color = if (vibration.active) OK else IDLE,
                )
                IndicatorRow(
                    title = when {
                        aiLatencyMs != null -> "AI Speed Model: Running"
                        nhc.phase == NhcStatus.Phase.OFF_NO_MODEL || nhc.phase == NhcStatus.Phase.UNAVAILABLE -> "AI Speed Model: Off"
                        else -> "AI Speed Model: Waiting"
                    },
                    detail = if (aiLatencyMs != null) "Model A speed estimate  ·  ${Display.ms(aiLatencyMs)} per inference"
                    else "Runs once the engine has aligned its heading",
                    color = if (aiLatencyMs != null) OK else IDLE,
                )
                val (nhcTitle, nhcDetail, nhcColor) = nhcText(nhc, deadReckoning)
                IndicatorRow(nhcTitle, nhcDetail, nhcColor)
                IndicatorRow(
                    title = when {
                        !hasMap -> "Map-Matching: No Road Map"
                        mapMatchConfidence != null -> "Map-Matching: Snapped to Road"
                        else -> "Map-Matching: Searching"
                    },
                    detail = when {
                        !hasMap -> "No offline road bundle loaded"
                        mapMatchConfidence != null ->
                            "${roadName ?: "Unnamed road"}  ·  ${Display.percent(mapMatchConfidence)} confidence (green track)"
                        else -> "No confident road match yet"
                    },
                    color = if (hasMap && mapMatchConfidence != null) OK else IDLE,
                )
            }
        }
        if (flashing) vibration.last?.let { ShockWarning(it.rawPeakDevMps2, it.averagedDevMps2, it.suppressed) }
    }
}

private val GLASS_TOP = Color(0xCC1C1C22)
private val GLASS_BOTTOM = Color(0xB30A0A0C)
private val OK = Color(0xFF00E676)
private val IDLE = Color(0xFF8A8A94)
private val ACTIVE_DR = RouteonOrange
private val MUTED = Color(0xFFB8B8C0)

/** NHC phase -> (title, detail, colour). "Applied" names dead reckoning only when it is happening. */
private fun nhcText(s: NhcStatus, deadReckoning: Boolean): Triple<String, String, Color> {
    val applied = "${s.acceptedTotal} constraint updates accepted"
    return when (s.phase) {
        NhcStatus.Phase.UNAVAILABLE -> Triple("NHC: Unavailable", "GNSS-only engine: no dead reckoning", IDLE)
        NhcStatus.Phase.OFF_NO_MODEL -> Triple("NHC: Off", "No speed model on this device: NHC is disabled with it", IDLE)
        NhcStatus.Phase.WAITING -> Triple("NHC Map-Matching: Ready", "Starts once the engine has locked its heading", IDLE)
        NhcStatus.Phase.APPLIED -> if (deadReckoning) {
            Triple("NHC Constraints Applied", "Holding the dead-reckoned velocity to the vehicle axis  ·  $applied", ACTIVE_DR)
        } else {
            Triple("NHC Constraints Applied", "Alongside GNSS  ·  $applied", OK)
        }
        NhcStatus.Phase.GATED -> Triple("NHC: Update Rejected", "Filter innovation gate refused the last constraint  ·  $applied", IDLE)
        NhcStatus.Phase.LOW_SPEED -> Triple("NHC Map-Matching: Ready", "Standing by below 2 m/s  ·  $applied", IDLE)
        NhcStatus.Phase.HARD_CORNERING -> Triple("NHC: Paused", "Hard cornering: constraint skipped  ·  $applied", IDLE)
    }
}

@Composable
private fun IndicatorRow(title: String, detail: String, color: Color) {
    Row(verticalAlignment = Alignment.Top) {
        Box(Modifier.padding(top = 5.dp).size(8.dp).background(color, CircleShape))
        Spacer(Modifier.width(10.dp))
        Column {
            Text(title, color = Color.White, fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.bodyMedium)
            Text(detail, color = MUTED, style = MaterialTheme.typography.bodySmall)
        }
    }
}

/** Pulsing warning for [com.sih26168.deadreckoning.app.SHOCK_FLASH_S] after a spike: raw peak -> value the INS got. */
@Composable
private fun ShockWarning(rawPeakMps2: Double, averagedMps2: Double, suppressed: Boolean) {
    val pulse = rememberInfiniteTransition(label = "shock")
    val a by pulse.animateFloat(0.55f, 1f, infiniteRepeatable(tween(250), RepeatMode.Reverse), label = "shock-alpha")
    val numbers = String.format(Locale.ROOT, "%.1f → %.1f m/s² after averaging", rawPeakMps2, averagedMps2)
    Column(
        Modifier
            .fillMaxWidth()
            .alpha(a)
            .background(Brush.horizontalGradient(listOf(Color(0xFFB0001E), ACTIVE_DR)), RoundedCornerShape(10.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Text(if (suppressed) "⚠ Pothole/Shock Suppressed" else "⚠ Shock Attenuated (still above threshold)",
            color = Color.White, fontWeight = FontWeight.ExtraBold, style = MaterialTheme.typography.bodyMedium)
        Text(numbers, color = Color.White, style = MaterialTheme.typography.bodySmall)
    }
}

/** Small status dot that breathes while live. */
@Composable
private fun LiveDot(color: Color) {
    val pulse = rememberInfiniteTransition(label = "live")
    val a by pulse.animateFloat(0.35f, 1f, infiniteRepeatable(tween(900), RepeatMode.Reverse), label = "live-alpha")
    Box(Modifier.size(8.dp).alpha(a).background(color, CircleShape))
}
