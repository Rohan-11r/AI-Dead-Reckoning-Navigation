package com.sih26168.deadreckoning.app.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.Crossfade
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BatteryChargingFull
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.MyLocation
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Update
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.sih26168.deadreckoning.app.ui.theme.RouteonRedDeep
import kotlinx.coroutines.delay

/**
 * Routeon onboarding: a short splash, then the user's name and every permission the navigation
 * service needs. [onComplete] fires only when the name is non-blank AND [RequirementStatus.allMet];
 * the status is re-read on every resume, because background location and the battery exemption
 * are granted on system settings pages, outside this activity.
 *
 * Pure UI: no navigation, filter or engine code is touched here.
 */
@Composable
fun OnboardingScreen(initialName: String, onComplete: (name: String) -> Unit) {
    var showSplash by rememberSaveable { mutableStateOf(true) }
    LaunchedEffect(Unit) {
        delay(SPLASH_MS)
        showSplash = false
    }
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Crossfade(targetState = showSplash, label = "onboarding") { splash ->
            if (splash) Splash() else Setup(initialName, onComplete)
        }
    }
}

/** What onboarding requires. The single definition used by both the screen and MainActivity's gate. */
data class RequirementStatus(
    val preciseLocation: Boolean,
    val backgroundLocation: Boolean,
    val highRateSensors: Boolean,
    val batteryUnrestricted: Boolean,
) {
    val allMet: Boolean get() = preciseLocation && backgroundLocation && highRateSensors && batteryUnrestricted

    companion object {
        fun of(ctx: Context) = RequirementStatus(
            preciseLocation = granted(ctx, Manifest.permission.ACCESS_FINE_LOCATION),
            // before Android 10 there is no separate background grant: foreground location covers it
            backgroundLocation = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                granted(ctx, Manifest.permission.ACCESS_BACKGROUND_LOCATION)
            } else {
                granted(ctx, Manifest.permission.ACCESS_FINE_LOCATION)
            },
            // a NORMAL (install-time) permission from Android 12: declared in the manifest, never prompted
            highRateSensors = Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
                granted(ctx, Manifest.permission.HIGH_SAMPLING_RATE_SENSORS),
            batteryUnrestricted = ctx.getSystemService(PowerManager::class.java)
                ?.isIgnoringBatteryOptimizations(ctx.packageName) == true,
        )

        private fun granted(ctx: Context, permission: String) =
            ContextCompat.checkSelfPermission(ctx, permission) == PackageManager.PERMISSION_GRANTED
    }
}

private const val SPLASH_MS = 1_200L
private const val NAME_MAX_CHARS = 40

@Composable
private fun Splash() {
    val scheme = MaterialTheme.colorScheme
    Column(
        Modifier
            .fillMaxSize()
            // faint red glow rising from the centre of the black background
            .background(Brush.radialGradient(listOf(scheme.primary.copy(alpha = 0.22f), Color.Transparent))),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        RouteonLogo(size = 132)
        Spacer(Modifier.height(24.dp))
        Text("Routeon", style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Bold,
            color = scheme.onBackground)
        Spacer(Modifier.height(8.dp))
        Text(
            "Keeps navigating when GNSS drops",
            style = MaterialTheme.typography.bodyLarge,
            color = scheme.onSurfaceVariant,
        )
    }
}

/** Compass disc with a car badge -- standard Material icons only. */
@Composable
private fun RouteonLogo(size: Int) {
    val scheme = MaterialTheme.colorScheme
    Box(Modifier.size(size.dp), contentAlignment = Alignment.Center) {
        Box(
            Modifier
                .shadow(elevation = (size * 0.18f).dp, shape = CircleShape,
                    ambientColor = scheme.primary, spotColor = scheme.primary)
                .size(size.dp)
                .background(Brush.linearGradient(listOf(scheme.primary, RouteonRedDeep)), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Filled.Explore, contentDescription = null, tint = scheme.onPrimary,
                modifier = Modifier.size((size * 0.62f).dp))
        }
        Box(
            Modifier
                .align(Alignment.BottomEnd)
                .size((size * 0.36f).dp)
                .background(scheme.background, CircleShape)
                .padding(4.dp)
                .background(scheme.primaryContainer, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Filled.DirectionsCar, contentDescription = null, tint = scheme.onPrimaryContainer,
                modifier = Modifier.size((size * 0.2f).dp))
        }
    }
}

@Composable
private fun Setup(initialName: String, onComplete: (String) -> Unit) {
    val context = LocalContext.current
    var name by rememberSaveable { mutableStateOf(initialName) }
    var status by remember { mutableStateOf(RequirementStatus.of(context)) }
    // after a denial the system may stop showing the dialog; the button then opens app settings
    var locationDenied by rememberSaveable { mutableStateOf(false) }
    var backgroundDenied by rememberSaveable { mutableStateOf(false) }
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { status = RequirementStatus.of(context) }

    val locationLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { r ->
        // Android 12+ lets the user pick "approximate": that grants COARSE only and is NOT enough
        locationDenied = r[Manifest.permission.ACCESS_FINE_LOCATION] != true
        status = RequirementStatus.of(context)
    }
    val backgroundLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        backgroundDenied = !ok
        status = RequirementStatus.of(context)
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        RouteonLogo(size = 72)
        Spacer(Modifier.height(16.dp))
        Text("Welcome to Routeon", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(6.dp))
        Text(
            "Routeon estimates your position from the phone's motion sensors when satellite signals are lost.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(24.dp))

        OutlinedTextField(
            value = name,
            onValueChange = { name = it.take(NAME_MAX_CHARS) },
            label = { Text("Your name") },
            leadingIcon = { Icon(Icons.Filled.Person, contentDescription = null) },
            singleLine = true,
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words, imeAction = ImeAction.Done),
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(24.dp))

        Text("Required permissions", style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground, modifier = Modifier.fillMaxWidth())
        Spacer(Modifier.height(8.dp))

        RequirementRow(
            icon = Icons.Filled.MyLocation,
            title = "Precise location",
            detail = "Exact GNSS fixes initialise and correct the dead-reckoning filter. \"Approximate\" is not enough.",
            met = status.preciseLocation,
            actionLabel = if (locationDenied) "Open settings" else "Allow",
            onAction = {
                if (locationDenied) openAppSettings(context)
                else locationLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION))
            },
        )
        RequirementRow(
            icon = Icons.Filled.Update,
            title = "Background location",
            detail = if (status.preciseLocation) "Choose \"Allow all the time\" so navigation continues with the screen off."
            else "Grant precise location first.",
            met = status.backgroundLocation,
            enabled = status.preciseLocation,
            actionLabel = if (backgroundDenied) "Open settings" else "Allow",
            onAction = {
                if (backgroundDenied || Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) openAppSettings(context)
                else backgroundLauncher.launch(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
            },
        )
        RequirementRow(
            icon = Icons.Filled.Speed,
            title = "High-frequency sensors",
            detail = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                if (status.highRateSensors) "Granted at install (Android 12+)."
                else "Not granted: reinstall the app to receive it."
            } else "Not required below Android 12.",
            met = status.highRateSensors,
            actionLabel = null,
            onAction = {},
        )
        RequirementRow(
            icon = Icons.Filled.BatteryChargingFull,
            title = "Disable battery optimization",
            detail = "Stops Android from pausing sensor recording during a drive.",
            met = status.batteryUnrestricted,
            actionLabel = "Disable",
            onAction = { requestBatteryExemption(context) },
        )

        Spacer(Modifier.height(24.dp))
        val ready = name.isNotBlank() && status.allMet
        Button(onClick = { onComplete(name.trim()) }, enabled = ready, modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text("Start navigating")
        }
        if (!ready) {
            Spacer(Modifier.height(8.dp))
            Text(
                if (name.isBlank()) "Enter your name and grant every permission to continue."
                else "Grant every permission above to continue.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun RequirementRow(
    icon: ImageVector,
    title: String,
    detail: String,
    met: Boolean,
    actionLabel: String?,
    onAction: () -> Unit,
    enabled: Boolean = true,
) {
    val scheme = MaterialTheme.colorScheme
    Card(
        Modifier.fillMaxWidth().padding(vertical = 4.dp),
        colors = CardDefaults.cardColors(containerColor = scheme.surfaceContainerHigh, contentColor = scheme.onSurface),
        // granted rows get a red edge; pending rows a neutral one
        border = BorderStroke(1.dp, if (met) scheme.primary else scheme.outline),
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = scheme.primary)
            Spacer(Modifier.width(16.dp))
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleSmall, color = scheme.onSurface)
                Text(detail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.width(12.dp))
            when {
                met -> Icon(Icons.Filled.CheckCircle, contentDescription = "Granted",
                    tint = MaterialTheme.colorScheme.primary)
                actionLabel != null -> OutlinedButton(
                    onClick = onAction,
                    enabled = enabled,
                    border = BorderStroke(1.dp, if (enabled) scheme.primary else scheme.outline),
                ) { Text(actionLabel) }
            }
        }
    }
}

private fun openAppSettings(ctx: Context) {
    ctx.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", ctx.packageName, null)))
}

@SuppressLint("BatteryLife") // continuous sensor acquisition is this app's core function
private fun requestBatteryExemption(ctx: Context) {
    try {
        ctx.startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:${ctx.packageName}")))
    } catch (e: ActivityNotFoundException) {
        // some OEM builds lack the direct dialog: fall back to the system list
        ctx.startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
    }
}
