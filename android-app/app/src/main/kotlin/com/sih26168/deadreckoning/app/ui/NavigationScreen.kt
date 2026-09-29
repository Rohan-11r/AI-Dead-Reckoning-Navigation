package com.sih26168.deadreckoning.app.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.DashPathEffect
import android.graphics.PorterDuff
import android.location.Location
import android.location.LocationManager
import android.view.MotionEvent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.sih26168.deadreckoning.app.AlignmentStatus
import com.sih26168.deadreckoning.app.GeoTrackPoint
import com.sih26168.deadreckoning.app.MODE_WAITING
import com.sih26168.deadreckoning.app.R
import com.sih26168.deadreckoning.app.gnssOutShown
import com.sih26168.deadreckoning.app.trackRuns
import com.sih26168.deadreckoning.app.ui.theme.RouteonOrange
import com.sih26168.deadreckoning.app.ui.theme.RouteonRed
import com.sih26168.deadreckoning.app.ui.theme.RouteonRedDeep
import com.sih26168.deadreckoning.core.Display
import com.sih26168.deadreckoning.core.GnssState
import com.sih26168.deadreckoning.core.NavSnapshot
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.CustomZoomButtonsController
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.FolderOverlay
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import org.osmdroid.views.overlay.mylocation.GpsMyLocationProvider
import org.osmdroid.views.overlay.mylocation.IMyLocationProvider
import org.osmdroid.views.overlay.mylocation.MyLocationNewOverlay

/**
 * Primary screen, laid out like a navigation app: the OpenStreetMap view fills the whole screen
 * (under the translucent top bar) and everything else floats on top of it -- the dead-reckoning
 * banner at the top, a bottom panel with speed, state, alignment, the "Simulate GNSS outage"
 * toggle and, on tap, the full HUD details.
 *
 * The map draws the ENGINE's (filter) track exactly as the engine reported it (degrees, no
 * re-projection) -- red where GNSS was fused, orange where the engine was dead reckoning --
 * the map-matched positions in green, and a navigation arrow at the latest position: icy
 * white-cyan while GNSS is fused, glowing orange with an "AI Dead Reckoning Active" banner while
 * the engine is dead reckoning (simulated outage OR real GNSS loss after initialisation, see
 * [gnssOutShown]). The arrow points along the displayed track ([displayBearingDeg]).
 *
 * Until the engine has a position the map centres on the DEVICE's own location (osmdroid
 * [MyLocationNewOverlay], raw platform location, display only -- never an engine input). That
 * dot is hidden during a simulated outage so the display does not show the withheld GNSS.
 * Tiles are display only and drawn colour-inverted ([DARK_TILES]) for the dark theme: with no
 * network the map is blank but the track, car and HUD keep updating.
 *
 * The flag + faint dashed line are a DEMO DESTINATION ([DEMO_DESTINATION]): a fixed,
 * display-only pin, labelled as such, never an input to the engine or map matching; the line
 * is a straight line, not a computed route.
 *
 * [contentPadding] is the Scaffold's (top bar, bottom navigation): the map ignores it and draws
 * underneath; the floating panels respect it.
 */
@Composable
fun NavigationScreen(
    snapshot: NavSnapshot?,
    geoTrack: List<GeoTrackPoint>,
    geoMatchedTrack: List<Pair<Double, Double>>,
    hasMap: Boolean,
    alignment: AlignmentStatus,
    outageOn: Boolean,
    running: Boolean,
    onOutage: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    contentPadding: PaddingValues = PaddingValues(0.dp),
) {
    val isDrEngine = snapshot != null && !snapshot.engineName.startsWith(GNSS_ONLY_ENGINE_PREFIX)
    val initialised = snapshot != null && snapshot.mode != MODE_WAITING
    val gnssOut = gnssOutShown(snapshot?.state, snapshot?.mode, outageOn, running)
    // screen space covered by the floating UI: the map centres the car in what is left visible
    var topInsetPx by remember { mutableIntStateOf(0) }
    var bottomInsetPx by remember { mutableIntStateOf(0) }
    Box(modifier.fillMaxSize()) {
        TrackMap(geoTrack, geoMatchedTrack, drActive = gnssOut, showDeviceLocation = !outageOn,
            insetTopPx = topInsetPx, insetBottomPx = bottomInsetPx, modifier = Modifier.fillMaxSize())
        Column(
            Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .onSizeChanged { topInsetPx = it.height }
                .padding(top = contentPadding.calculateTopPadding())
                .padding(12.dp),
        ) {
            if (gnssOut) DeadReckoningBanner(simulated = outageOn, isDrEngine = isDrEngine, initialised = initialised)
        }
        BottomPanel(
            snapshot, hasMap, alignment, outageOn, running, onOutage,
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .onSizeChanged { bottomInsetPx = it.height }
                .padding(bottom = contentPadding.calculateBottomPadding()),
        )
    }
}

/** Display-only demo destination: Manish Nagar underpass, Nagpur (approximate). */
private val DEMO_DESTINATION = GeoPoint(21.093, 79.068)
private const val GNSS_ONLY_ENGINE_PREFIX = "gnss-only" // GnssOnlyEngine.name: it cannot dead-reckon
private val CAR_GNSS = Color(0xFFBFF4FF) // icy white-cyan: clearly distinct from the orange dead-reckoning car
private val CAR_DR = RouteonOrange
private val NO_DR = RouteonRedDeep
private val TRACK = RouteonRedDeep // darker than the car, so the bright car stands out at the track head
private val TRACK_DR = CAR_DR // dead-reckoned stretches of the track
private val MATCHED = Color(0xFF00E676) // bright green: readable on the inverted (dark) tiles
private val LABEL_MUTED = Color(0xFFB8B8C0)
private const val FOLLOW_ZOOM = 17.0
private val PANEL_SHAPE = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
private const val PANEL_ALPHA = 0.94f // the map shows faintly through the panels

@Composable
private fun DeadReckoningBanner(simulated: Boolean, isDrEngine: Boolean, initialised: Boolean) {
    val (title, detail) = when {
        !isDrEngine -> "GNSS lost - no dead reckoning" to "GNSS-only engine is running: the position is not updating"
        !initialised -> "GNSS withheld - dead reckoning not started" to
            "The engine starts from a moving GNSS fix (> 5 m/s): turn the outage off to let it initialise"
        simulated -> "AI Dead Reckoning Active" to "GNSS withheld (simulated outage): position from IMU + AI speed"
        else -> "AI Dead Reckoning Active" to "GNSS signal lost: position from IMU + AI speed"
    }
    val glow = if (isDrEngine) CAR_DR else NO_DR
    val shape = RoundedCornerShape(12.dp)
    Column(
        Modifier
            .fillMaxWidth()
            // coloured shadow = the "glow" around the banner on the black background
            .shadow(elevation = 18.dp, shape = shape, ambientColor = glow, spotColor = glow)
            .background(
                if (isDrEngine) Brush.horizontalGradient(listOf(RouteonRed, CAR_DR)) else Brush.horizontalGradient(listOf(NO_DR, NO_DR)),
                shape,
            )
            .padding(horizontal = 16.dp, vertical = 10.dp),
    ) {
        Text(title, color = Color.White, fontWeight = FontWeight.ExtraBold, style = MaterialTheme.typography.titleMedium)
        Text(detail, color = Color.White, style = MaterialTheme.typography.bodySmall)
    }
}

/**
 * Floating bottom panel: speed + GNSS state, heading alignment and the outage toggle always;
 * road / map match / engine / mode behind the grab handle, so the map stays mostly visible.
 */
@Composable
private fun BottomPanel(
    s: NavSnapshot?,
    hasMap: Boolean,
    alignment: AlignmentStatus,
    outageOn: Boolean,
    running: Boolean,
    onOutage: (Boolean) -> Unit,
    modifier: Modifier,
) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    Column(
        modifier
            .shadow(elevation = 24.dp, shape = PANEL_SHAPE)
            .background(MaterialTheme.colorScheme.surface.copy(alpha = PANEL_ALPHA), PANEL_SHAPE)
            .padding(start = 16.dp, end = 16.dp, bottom = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .clickable(onClickLabel = if (expanded) "Hide details" else "Show details") { expanded = !expanded }
                .padding(top = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Box(Modifier.size(width = 40.dp, height = 4.dp).background(LABEL_MUTED.copy(alpha = 0.6f), RoundedCornerShape(2.dp)))
            Text(if (expanded) "Hide details" else "Details", style = MaterialTheme.typography.labelSmall,
                color = LABEL_MUTED, modifier = Modifier.padding(top = 4.dp))
        }
        SpeedRow(s, outageOn)
        AlignmentBar(alignment)
        AnimatedVisibility(visible = expanded) { Details(s, hasMap) }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Simulate GNSS outage", fontWeight = FontWeight.Bold, color = Color.White)
                Text("Withholds every fix from the engine; still logged, flagged withheld_by_sim",
                    style = MaterialTheme.typography.bodySmall, color = LABEL_MUTED)
            }
            Switch(checked = outageOn, onCheckedChange = onOutage, enabled = running)
        }
    }
}

@Composable
private fun SpeedRow(s: NavSnapshot?, outageOn: Boolean) {
    val stateColor = when (s?.state) {
        GnssState.GOOD -> Color(0xFF2E7D32)
        GnssState.DEGRADED -> Color(0xFFF9A825)
        GnssState.RECOVERING -> Color(0xFF1565C0)
        GnssState.LOST -> Color(0xFFC62828)
        null -> LABEL_MUTED
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(Display.speedKmh(s?.speedMps), style = MaterialTheme.typography.displaySmall,
                fontWeight = FontWeight.Bold, color = Color.White)
            Text("Confidence (within 10 m) ${Display.percent(s?.confidence)}  ·  σ ${Display.metres(s?.sigmaHm)}",
                style = MaterialTheme.typography.bodySmall, color = LABEL_MUTED)
        }
        Text(Display.stateLabel(s?.state, outageOn), color = Color.White, fontWeight = FontWeight.Bold,
            style = MaterialTheme.typography.labelMedium,
            modifier = Modifier.background(stateColor, RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 6.dp))
    }
}

/** The engine's heading alignment ([AlignmentStatus]): locked = the engine chose its heading. */
@Composable
private fun AlignmentBar(a: AlignmentStatus) {
    if (a.phase == AlignmentStatus.Phase.UNAVAILABLE) return
    val (label, color) = when (a.phase) {
        AlignmentStatus.Phase.WAITING_FOR_GNSS -> "Heading alignment: waiting for GNSS at > 5 m/s" to LABEL_MUTED
        AlignmentStatus.Phase.ALIGNING -> "Heading alignment: ${a.percent} %  (keep GNSS until locked)" to CAR_DR
        else -> "Heading locked: dead reckoning ready" to MATCHED
    }
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(label, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold, color = Color.White)
        LinearProgressIndicator(progress = { a.percent / 100f }, color = color,
            trackColor = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth())
    }
}

@Composable
private fun Details(s: NavSnapshot?, hasMap: Boolean) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        HudRow("Road", Display.roadName(s?.roadName, hasMap))
        HudRow("Map match (confidence)", Display.percent(s?.mapMatchConfidence))
        HudRow("Engine", s?.engineName ?: Display.NA)
        HudRow("Mode", s?.mode ?: Display.NA)
        s?.note?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
    }
}

@Composable
internal fun HudRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium, color = LABEL_MUTED)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, color = Color.White)
    }
}

/**
 * Dark-mode tiles: invert RGB (c' = 255 - c), keep alpha. Applied to the tiles overlay only, so
 * the track, matched line, flag and car keep their true colours on top of the dark map.
 */
private val DARK_TILES = ColorMatrixColorFilter(
    ColorMatrix(
        floatArrayOf(
            -1f, 0f, 0f, 0f, 255f,
            0f, -1f, 0f, 0f, 255f,
            0f, 0f, -1f, 0f, 255f,
            0f, 0f, 0f, 1f, 0f,
        ),
    ),
)

/** Fine or coarse location granted: the device-location dot and initial centre need one. */
private fun hasLocationPermission(ctx: Context): Boolean =
    listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        .any { ContextCompat.checkSelfPermission(ctx, it) == PackageManager.PERMISSION_GRANTED }

/** The freshest platform last-known location, for the map's first centre only (display only). */
@SuppressLint("MissingPermission") // guarded by hasLocationPermission
private fun lastKnownLocation(ctx: Context): GeoPoint? {
    if (!hasLocationPermission(ctx)) return null
    val lm = ctx.getSystemService(LocationManager::class.java) ?: return null
    return lm.getProviders(true)
        .mapNotNull { provider ->
            try {
                lm.getLastKnownLocation(provider)
            } catch (x: SecurityException) { // permission revoked between the check and the call
                null
            }
        }
        .maxByOrNull { it.elapsedRealtimeNanos }
        ?.let { GeoPoint(it.latitude, it.longitude) }
}

/**
 * osmdroid's device-location dot, also handing every location to [onFix]. Raw platform
 * location, display only: it centres the map, it never reaches the engine.
 */
private class DeviceLocationOverlay(ctx: Context, map: MapView, private val onFix: (GeoPoint) -> Unit) :
    MyLocationNewOverlay(GpsMyLocationProvider(ctx), map) {
    override fun onLocationChanged(location: Location?, source: IMyLocationProvider?) {
        super.onLocationChanged(location, source)
        if (location != null) onFix(GeoPoint(location.latitude, location.longitude))
    }
}

/** Overlays created once per MapView; [TrackMap] only moves them. */
private class MapOverlays(private val map: MapView, ctx: Context, onDeviceFix: (GeoPoint) -> Unit) {
    val demoLine = Polyline(map).apply {
        outlinePaint.color = LABEL_MUTED.copy(alpha = 0.55f).toArgb()
        outlinePaint.strokeWidth = 5f
        outlinePaint.pathEffect = DashPathEffect(floatArrayOf(24f, 16f), 0f)
        title = "Demo destination - straight line, not a route"
    }
    val matched = Polyline(map).apply {
        outlinePaint.color = MATCHED.toArgb()
        outlinePaint.strokeWidth = 6f
    }

    /** The engine track, one Polyline per GNSS / dead-reckoning run ([trackRuns]). */
    private val track = FolderOverlay()
    private val segments = mutableListOf<Polyline>()
    val destination = Marker(map).apply {
        position = DEMO_DESTINATION
        icon = ContextCompat.getDrawable(ctx, R.drawable.ic_map_flag)
        setAnchor(0.2f, 1.0f) // the foot of the flag pole
        title = "Demo destination: Manish Nagar underpass (display only)"
    }
    private val device = DeviceLocationOverlay(ctx, map, onDeviceFix)
    val car = Marker(map).apply {
        // greyscale + MULTIPLY tint: the state colour keeps the arrow's bevel and shadow (see the XML)
        icon = ContextCompat.getDrawable(ctx, R.drawable.ic_premium_car)?.mutate()?.apply {
            setTintMode(PorterDuff.Mode.MULTIPLY)
        }
        setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        isFlat = true // rotation is relative to the map, not the screen
        title = "Engine position"
        setInfoWindow(null)
    }

    var resumed = false
    var deviceWanted = false
    private var carTint: Int? = null
    private var centerOffsetY = 0

    /** Arrow colour for the engine's mode; re-tints only on a change. */
    fun tintCar(argb: Int) {
        if (argb == carTint) return
        car.icon?.setTint(argb)
        carTint = argb
    }

    /** Arrow direction; with no bearing (not moving far enough) it keeps the last one. */
    fun pointCar(bearingDeg: Double?) {
        // osmdroid's Marker rotation is counter-clockwise in degrees; a bearing is clockwise from north
        if (bearingDeg != null) car.rotation = -bearingDeg.toFloat()
    }

    /** Puts the map's centre in the middle of what the floating panels leave visible. */
    fun setCenterOffset(insetTopPx: Int, insetBottomPx: Int) {
        val y = (insetTopPx - insetBottomPx) / 2
        if (y == centerOffsetY) return
        map.setMapCenterOffset(0, y)
        centerOffsetY = y
    }

    init {
        // bottom to top: demo line, matched, track, flag, device dot, car
        map.overlays.addAll(listOf(demoLine, matched, track, destination, device, car))
    }

    fun setTrack(points: List<GeoTrackPoint>) {
        val runs = trackRuns(points)
        while (segments.size > runs.size) track.remove(segments.removeAt(segments.lastIndex))
        while (segments.size < runs.size) {
            val line = Polyline(map).apply { outlinePaint.strokeWidth = 9f }
            segments += line
            track.add(line)
        }
        runs.forEachIndexed { i, (deadReckoning, pts) ->
            segments[i].outlinePaint.color = (if (deadReckoning) TRACK_DR else TRACK).toArgb()
            segments[i].setPoints(pts.map { GeoPoint(it.latDeg, it.lonDeg) })
        }
    }

    /**
     * Runs the device-location provider only while the map is resumed, the dot is wanted and
     * location is granted. Compares with the overlay's own flag, which osmdroid's onPause/onResume also toggle.
     */
    @SuppressLint("MissingPermission") // guarded by hasLocationPermission
    fun syncDevice() {
        val on = resumed && deviceWanted && hasLocationPermission(map.context)
        if (on == device.isMyLocationEnabled) return
        if (on) device.enableMyLocation() else device.disableMyLocation()
    }
}

@SuppressLint("ClickableViewAccessibility") // the touch listener only turns auto-follow off; the map handles the gesture
@Composable
private fun TrackMap(
    geoTrack: List<GeoTrackPoint>,
    geoMatched: List<Pair<Double, Double>>,
    drActive: Boolean,
    showDeviceLocation: Boolean,
    insetTopPx: Int,
    insetBottomPx: Int,
    modifier: Modifier,
) {
    val context = LocalContext.current
    val insetBottom = with(LocalDensity.current) { insetBottomPx.toDp() }
    var follow by rememberSaveable { mutableStateOf(true) }
    // latest device location (display only): the map follows it until the engine has a position
    var deviceFix by remember { mutableStateOf<GeoPoint?>(null) }
    val mapView = remember {
        // OSM tile usage policy: identify the app; tiles cache in app-private storage
        Configuration.getInstance().load(context, context.getSharedPreferences("osmdroid", Context.MODE_PRIVATE))
        Configuration.getInstance().userAgentValue = context.packageName
        MapView(context).apply {
            setTileSource(TileSourceFactory.MAPNIK)
            overlayManager.tilesOverlay.setColorFilter(DARK_TILES)
            // placeholder grid while tiles load / with no network: black, not osmdroid's light grey
            overlayManager.tilesOverlay.setLoadingBackgroundColor(android.graphics.Color.BLACK)
            overlayManager.tilesOverlay.setLoadingLineColor(android.graphics.Color.rgb(0x2A, 0x2A, 0x2E))
            setBackgroundColor(android.graphics.Color.BLACK)
            setMultiTouchControls(true)
            zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER) // pinch to zoom: no +/- clutter
            controller.setZoom(FOLLOW_ZOOM)
            // the demo pin only when the device has never had a location (or it is not granted)
            controller.setCenter(lastKnownLocation(context) ?: DEMO_DESTINATION)
            setOnTouchListener { _, e ->
                if (e.action == MotionEvent.ACTION_DOWN) follow = false
                false
            }
        }
    }
    val overlays = remember(mapView) {
        MapOverlays(mapView, context) { p -> mapView.post { deviceFix = p } } // provider thread -> main
    }

    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle, mapView) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> {
                    mapView.onResume()
                    overlays.resumed = true
                    overlays.syncDevice()
                }
                Lifecycle.Event.ON_PAUSE -> {
                    overlays.resumed = false
                    overlays.syncDevice()
                    mapView.onPause()
                }
                else -> Unit
            }
        }
        lifecycle.addObserver(observer)
        onDispose {
            lifecycle.removeObserver(observer)
            mapView.onDetach() // detaches every overlay: stops the device-location provider too
        }
    }

    Box(modifier) {
        AndroidView(factory = { mapView }, modifier = Modifier.fillMaxSize(), update = { map ->
            overlays.setCenterOffset(insetTopPx, insetBottomPx)
            overlays.deviceWanted = showDeviceLocation
            overlays.syncDevice()
            overlays.setTrack(geoTrack)
            overlays.matched.setPoints(geoMatched.map { (lat, lon) -> GeoPoint(lat, lon) })
            val start = geoTrack.firstOrNull()?.let { GeoPoint(it.latDeg, it.lonDeg) }
            overlays.demoLine.setPoints(if (start != null) listOf(start, DEMO_DESTINATION) else emptyList())
            val last = geoTrack.lastOrNull()
            overlays.car.setVisible(last != null)
            if (last != null) {
                val here = GeoPoint(last.latDeg, last.lonDeg)
                overlays.car.position = here
                overlays.tintCar((if (drActive) CAR_DR else CAR_GNSS).toArgb())
                overlays.pointCar(displayBearingDeg(geoTrack))
                if (follow) map.controller.setCenter(here)
            } else if (follow) {
                deviceFix?.let { map.controller.setCenter(it) } // engine not initialised yet: follow the device
            }
            map.invalidate()
        })
        if (!follow) {
            FilledTonalButton(onClick = { follow = true },
                modifier = Modifier.align(Alignment.BottomEnd).padding(bottom = insetBottom).padding(12.dp)) {
                Text("Recenter")
            }
        }
        Text(
            "© OpenStreetMap contributors",
            style = MaterialTheme.typography.labelSmall,
            color = LABEL_MUTED,
            modifier = Modifier.align(Alignment.BottomStart).padding(bottom = insetBottom)
                .background(Color.Black.copy(alpha = 0.7f)).padding(4.dp),
        )
    }
}

private const val HEADING_MIN_M = 3.0 // below this the direction is position noise, not travel
private const val HEADING_LOOKBACK = 50 // track points (5 s at the 10 Hz publish rate)

/**
 * Direction of travel for the arrow, degrees clockwise from north: from the most recent track
 * point at least [HEADING_MIN_M] behind the head to the head, within the last
 * [HEADING_LOOKBACK] points; null when the car has not moved that far. Display only: derived
 * from the engine's own positions, never fed back.
 */
private fun displayBearingDeg(track: List<GeoTrackPoint>): Double? {
    val head = track.lastOrNull() ?: return null
    val here = GeoPoint(head.latDeg, head.lonDeg)
    for (i in track.size - 2 downTo maxOf(0, track.size - 1 - HEADING_LOOKBACK)) {
        val p = GeoPoint(track[i].latDeg, track[i].lonDeg)
        if (p.distanceToAsDouble(here) >= HEADING_MIN_M) return p.bearingTo(here)
    }
    return null
}
