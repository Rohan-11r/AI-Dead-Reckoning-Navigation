package com.sih26168.deadreckoning.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

/**
 * Routeon theme: always dark (deep black background, red accents), whatever the system setting.
 * Every `on*` colour is white or light grey so text stays readable on the black surfaces.
 */
private val RouteonColors = darkColorScheme(
    primary = RouteonRed,
    onPrimary = RouteonOnDark,
    primaryContainer = RouteonRedContainer,
    onPrimaryContainer = RouteonOnDark,
    inversePrimary = RouteonRedDeep,
    secondary = RouteonRed,
    onSecondary = RouteonOnDark,
    secondaryContainer = RouteonRedContainer,
    onSecondaryContainer = RouteonOnDark,
    tertiary = RouteonOrange,
    onTertiary = RouteonOnDark,
    tertiaryContainer = RouteonRedContainer,
    onTertiaryContainer = RouteonOnDark,
    background = RouteonBlack,
    onBackground = RouteonOnDark,
    surface = RouteonSurface,
    onSurface = RouteonOnDark,
    surfaceVariant = RouteonSurfaceHigh,
    onSurfaceVariant = RouteonOnDarkMuted,
    surfaceTint = RouteonRed,
    surfaceContainerLowest = RouteonBlack,
    surfaceContainerLow = RouteonSurface,
    surfaceContainer = RouteonSurface,
    surfaceContainerHigh = RouteonSurfaceHigh,
    surfaceContainerHighest = RouteonSurfaceHighest,
    outline = RouteonOutline,
    outlineVariant = RouteonOutline,
    error = RouteonError,
    onError = RouteonBlack,
)

@Composable
fun RouteonTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = RouteonColors, content = content)
}
