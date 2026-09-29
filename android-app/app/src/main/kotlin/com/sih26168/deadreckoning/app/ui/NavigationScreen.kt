package com.sih26168.deadreckoning.app.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.DashPathEffect
import android.view.MotionEvent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.sih26168.deadreckoning.app.AlignmentStatus
import com.sih26168.deadreckoning.app.R
import com.sih26168.deadreckoning.core.Display
import com.sih26168.deadreckoning.core.GnssState
import com.sih26168.deadreckoning.core.NavSnapshot
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

/**
 * Primary screen: OpenStreetMap view + telemetry HUD + "Simulate GNSS outage" toggle.
 * The map draws the ENGINE's (filter) track exactly as the engine reported it (degrees, no
 * re-projection), the map-matched positions in green, and a car at the latest position:
 * blue while GNSS is fused, orange with an "AI Dead Reckoning Active" banner while the engine
 * is dead reckoning (simulated outage OR real GNSS loss). Tiles are display only: with no
 * network the map is blank but the track, car and HUD keep updating.
 *
 * The flag + faint dashed line are a DEMO DESTINATION ([DEMO_DESTINATION]): a fixed,
 * display-only pin, labelled as such, never an input to the engine or map matching; the line
 * is a straight line, not a computed route.
 */
@Composable
fun NavigationScreen(
    snapshot: NavSnapshot?,
    geoTrack: List<Pair<Double, Double>>,
    geoMatchedTrack: List<Pair<Double, Double>>,
    hasMap: Boolean,
    alignment: AlignmentStatus,
    outageOn: Boolean,
    running: Boolean,
    onOutage: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val isDrEngine = snapshot != null && !snapshot.engineName.startsWith(GNSS_ONLY_ENGINE_PREFIX)
    val gnssOut = outageOn || snapshot?.state == GnssState.LOST
    Column(modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (gnssOut) DeadReckoningBanner(simulated = outageOn, isDrEngine = isDrEngine)
        Card(Modifier.fillMaxWidth().weight(1f)) {
            TrackMap(geoTrack, geoMatchedTrack, drActive = gnssOut, modifier = Modifier.fillMaxSize())
        }
        AlignmentBar(alignment)
        Hud(snapshot, outageOn, hasMap)
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

/** Display-only demo destination: Manish Nagar underpass, Nagpur (approximate). */
private val DEMO_DESTINATION = GeoPoint(21.093, 79.068)
private const val GNSS_ONLY_ENGINE_PREFIX = "gnss-only" // GnssOnlyEngine.name: it cannot dead-reckon
private val CAR_GNSS = Color(0xFF1565C0)
private val CAR_DR = Color(0xFFEF6C00)
private val NO_DR = Color(0xFFC62828)
private val TRACK = Color(0xFF1565C0)
private val MATCHED = Color(0xFF2E7D32)
private const val FOLLOW_ZOOM = 17.0

@Composable
private fun DeadReckoningBanner(simulated: Boolean, isDrEngine: Boolean) {
    val (title, detail) = when {
        !isDrEngine -> "GNSS lost - no dead reckoning" to "GNSS-only engine is running: the position is not updating"
        simulated -> "AI Dead Reckoning Active" to "GNSS withheld (simulated outage): position from IMU + AI speed"
        else -> "AI Dead Reckoning Active" to "GNSS signal lost: position from IMU + AI speed"
    }
    Column(
        Modifier
            .fillMaxWidth()
            .background(if (isDrEngine) CAR_DR else NO_DR, RoundedCornerShape(12.dp))
            .padding(horizontal = 16.dp, vertical = 10.dp),
    ) {
        Text(title, color = Color.White, fontWeight = FontWeight.ExtraBold, style = MaterialTheme.typography.titleMedium)
        Text(detail, color = Color.White, style = MaterialTheme.typography.bodySmall)
    }
}

/** The engine's heading alignment ([AlignmentStatus]): locked = the engine chose its heading. */
@Composable
private fun AlignmentBar(a: AlignmentStatus) {
    if (a.phase == AlignmentStatus.Phase.UNAVAILABLE) return
    val (label, color) = when (a.phase) {
        AlignmentStatus.Phase.WAITING_FOR_GNSS -> "Heading alignment: waiting for GNSS at > 5 m/s" to Color.Gray
        AlignmentStatus.Phase.ALIGNING -> "Heading alignment: ${a.percent} %  (keep GNSS until locked)" to CAR_DR
        else -> "Heading locked: dead reckoning ready" to MATCHED
    }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(label, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
            LinearProgressIndicator(progress = { a.percent / 100f }, color = color, modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun Hud(s: NavSnapshot?, outageOn: Boolean, hasMap: Boolean) {
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
            HudRow("Road", Display.roadName(s?.roadName, hasMap))
            HudRow("Map match (confidence)", Display.percent(s?.mapMatchConfidence))
            HudRow("Engine", s?.engineName ?: Display.NA)
            HudRow("Mode", s?.mode ?: Display.NA)
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

/** Overlays created once per MapView; [TrackMap] only moves them. */
private class MapOverlays(map: MapView, ctx: Context) {
    val demoLine = Polyline(map).apply {
        outlinePaint.color = Color.Gray.copy(alpha = 0.55f).toArgb()
        outlinePaint.strokeWidth = 5f
        outlinePaint.pathEffect = DashPathEffect(floatArrayOf(24f, 16f), 0f)
        title = "Demo destination - straight line, not a route"
    }
    val matched = Polyline(map).apply {
        outlinePaint.color = MATCHED.toArgb()
        outlinePaint.strokeWidth = 6f
    }
    val track = Polyline(map).apply {
        outlinePaint.color = TRACK.toArgb()
        outlinePaint.strokeWidth = 9f
    }
    val destination = Marker(map).apply {
        position = DEMO_DESTINATION
        icon = ContextCompat.getDrawable(ctx, R.drawable.ic_map_flag)
        setAnchor(0.2f, 1.0f) // the foot of the flag pole
        title = "Demo destination: Manish Nagar underpass (display only)"
    }
    val car = Marker(map).apply {
        icon = ContextCompat.getDrawable(ctx, R.drawable.ic_map_car)?.mutate()
        setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        title = "Engine position"
        setInfoWindow(null)
    }

    init {
        // bottom to top: demo line, matched, track, flag, car
        map.overlays.addAll(listOf(demoLine, matched, track, destination, car))
    }
}

@SuppressLint("ClickableViewAccessibility") // the touch listener only turns auto-follow off; the map handles the gesture
@Composable
private fun TrackMap(
    geoTrack: List<Pair<Double, Double>>,
    geoMatched: List<Pair<Double, Double>>,
    drActive: Boolean,
    modifier: Modifier,
) {
    val context = LocalContext.current
    var follow by rememberSaveable { mutableStateOf(true) }
    val mapView = remember {
        // OSM tile usage policy: identify the app; tiles cache in app-private storage
        Configuration.getInstance().load(context, context.getSharedPreferences("osmdroid", Context.MODE_PRIVATE))
        Configuration.getInstance().userAgentValue = context.packageName
        MapView(context).apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(FOLLOW_ZOOM)
            controller.setCenter(DEMO_DESTINATION)
            setOnTouchListener { _, e ->
                if (e.action == MotionEvent.ACTION_DOWN) follow = false
                false
            }
        }
    }
    val overlays = remember(mapView) { MapOverlays(mapView, context) }

    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle, mapView) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> mapView.onResume()
                Lifecycle.Event.ON_PAUSE -> mapView.onPause()
                else -> Unit
            }
        }
        lifecycle.addObserver(observer)
        onDispose {
            lifecycle.removeObserver(observer)
            mapView.onDetach()
        }
    }

    Box(modifier) {
        AndroidView(factory = { mapView }, modifier = Modifier.fillMaxSize(), update = { map ->
            val pts = geoTrack.map { (lat, lon) -> GeoPoint(lat, lon) }
            overlays.track.setPoints(pts)
            overlays.matched.setPoints(geoMatched.map { (lat, lon) -> GeoPoint(lat, lon) })
            val start = pts.firstOrNull()
            overlays.demoLine.setPoints(if (start != null) listOf(start, DEMO_DESTINATION) else emptyList())
            val here = pts.lastOrNull()
            overlays.car.setVisible(here != null)
            if (here != null) {
                overlays.car.position = here
                overlays.car.icon?.setTint((if (drActive) CAR_DR else CAR_GNSS).toArgb())
                if (follow) map.controller.setCenter(here)
            }
            map.invalidate()
        })
        if (!follow) {
            FilledTonalButton(onClick = { follow = true }, modifier = Modifier.align(Alignment.BottomEnd).padding(12.dp)) {
                Text("Recenter")
            }
        }
        Text(
            "© OpenStreetMap contributors",
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.align(Alignment.BottomStart).background(Color.White.copy(alpha = 0.7f)).padding(4.dp),
        )
    }
}
