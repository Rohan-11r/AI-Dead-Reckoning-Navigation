package com.sih26168.deadreckoning.app.ui

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Route
import androidx.compose.material.icons.filled.SatelliteAlt
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.sih26168.deadreckoning.app.ui.theme.RouteonRedDeep

/**
 * Between the splash and "Enter your name": what Routeon does, in the dark/red theme, with an
 * animated "Next" arrow and a prominent "Start" button. Both call [onStart] (-> name screen).
 * Describes the pipeline only; it makes no accuracy claim.
 */
@Composable
fun WelcomeTransitionScreen(onStart: () -> Unit) {
    val scheme = MaterialTheme.colorScheme
    Column(
        Modifier
            .fillMaxSize()
            .background(scheme.background)
            // red glow falling from the top, fading into the black
            .background(Brush.verticalGradient(0f to scheme.primary.copy(alpha = 0.30f), 0.55f to Color.Transparent))
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 40.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        RouteonLogo(size = 96)
        Spacer(Modifier.height(28.dp))
        Text(
            "Navigation that keeps going",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.ExtraBold,
            color = scheme.onBackground,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(10.dp))
        Text(
            "Tunnels, underpasses, dense streets: when satellite signals drop, Routeon keeps estimating " +
                "your position from the phone's motion sensors.",
            style = MaterialTheme.typography.bodyLarge,
            color = scheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(32.dp))
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Feature(Icons.Filled.SatelliteAlt, "GNSS when it is there", "Satellite fixes start and correct the estimate")
            Feature(Icons.Filled.Sensors, "Sensors + AI when it is not", "Inertial dead reckoning with a learned speed estimate")
            Feature(Icons.Filled.Route, "Held to the road", "Map matching refines the track on the road network")
        }
        Spacer(Modifier.height(36.dp))
        NextArrow(onClick = onStart)
        Spacer(Modifier.height(28.dp))
        StartButton(onClick = onStart)
    }
}

@Composable
private fun Feature(icon: ImageVector, title: String, detail: String) {
    val scheme = MaterialTheme.colorScheme
    Row(
        Modifier
            .fillMaxWidth()
            .background(scheme.surfaceContainerHigh.copy(alpha = 0.85f), RoundedCornerShape(16.dp))
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier.size(40.dp).background(scheme.primary.copy(alpha = 0.16f), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(icon, contentDescription = null, tint = scheme.primary, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.width(14.dp))
        Column {
            Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold, color = scheme.onSurface)
            Text(detail, style = MaterialTheme.typography.bodySmall, color = scheme.onSurfaceVariant)
        }
    }
}

/** Glowing red disc with an arrow that nudges forward, inviting the tap. */
@Composable
private fun NextArrow(onClick: () -> Unit) {
    val scheme = MaterialTheme.colorScheme
    val pulse = rememberInfiniteTransition(label = "next-arrow")
    val nudge by pulse.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(900, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "nudge",
    )
    Box(
        Modifier
            .shadow(elevation = (10 + 10 * nudge).dp, shape = CircleShape, ambientColor = scheme.primary, spotColor = scheme.primary)
            .size(68.dp)
            .clip(CircleShape)
            .background(Brush.linearGradient(listOf(scheme.primary, RouteonRedDeep)))
            .clickable(role = Role.Button, onClickLabel = "Next", onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            Icons.AutoMirrored.Filled.ArrowForward,
            contentDescription = "Next",
            tint = scheme.onPrimary,
            modifier = Modifier.size(32.dp).graphicsLayer { translationX = nudge * 6.dp.toPx() },
        )
    }
}

/** Full-width red-gradient "Start" button. */
@Composable
private fun StartButton(onClick: () -> Unit) {
    val scheme = MaterialTheme.colorScheme
    val shape = RoundedCornerShape(28.dp)
    Row(
        Modifier
            .fillMaxWidth()
            .height(58.dp)
            .shadow(elevation = 16.dp, shape = shape, ambientColor = scheme.primary, spotColor = scheme.primary)
            .clip(shape)
            .background(Brush.horizontalGradient(listOf(RouteonRedDeep, scheme.primary)))
            .clickable(role = Role.Button, onClick = onClick),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("Start", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = scheme.onPrimary)
        Spacer(Modifier.width(10.dp))
        Icon(Icons.AutoMirrored.Filled.ArrowForward, contentDescription = null, tint = scheme.onPrimary)
    }
}
