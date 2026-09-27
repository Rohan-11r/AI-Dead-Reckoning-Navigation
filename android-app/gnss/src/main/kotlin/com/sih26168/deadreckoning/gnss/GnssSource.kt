package com.sih26168.deadreckoning.gnss

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.HandlerThread
import android.os.SystemClock
import androidx.annotation.RequiresPermission
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.sih26168.deadreckoning.core.GnssSample
import com.sih26168.deadreckoning.core.LocationMapping
import com.sih26168.deadreckoning.core.RawLocation
import com.sih26168.deadreckoning.core.mapLocation
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow

/** A mapped fix, or a location Android delivered that the schema rejected (with why). */
sealed interface GnssEvent {
    val provider: String

    data class Fix(val fix: GnssSample, override val provider: String) : GnssEvent
    data class Rejected(val reason: String, override val provider: String) : GnssEvent
}

interface GnssSource {
    val name: String
    fun events(): Flow<GnssEvent>
}

/** android.location.Location -> RawLocation (field copy only; the rules live in :core). */
fun Location.toRaw(): RawLocation = RawLocation(
    provider = provider ?: "unknown",
    latitudeDeg = latitude,
    longitudeDeg = longitude,
    altitudeM = if (hasAltitude()) altitude else null,
    accuracyM = if (hasAccuracy()) accuracy.toDouble() else null,
    speedMps = if (hasSpeed()) speed.toDouble() else null,
    bearingDeg = if (hasBearing()) bearing.toDouble() else null,
    elapsedRealtimeNs = elapsedRealtimeNanos,
    utcTimeMs = time,
    satellitesUsed = null, // needs a GnssStatus callback: later phase
)

internal fun toEvent(loc: Location, sessionStartNs: Long): GnssEvent {
    val received = SystemClock.elapsedRealtimeNanos()
    val p = loc.provider ?: "unknown"
    return when (val m = mapLocation(loc.toRaw(), sessionStartNs, received)) {
        is LocationMapping.Ok -> GnssEvent.Fix(m.fix, p)
        is LocationMapping.Rejected -> GnssEvent.Rejected(m.reason, p)
    }
}

/**
 * The platform GPS provider (default). It reports GNSS-derived fixes only -- no Wi-Fi or
 * cell-tower positions mixed in -- which is what IO-VNBD's phone recorded, and what an
 * honest outage demo needs: when the sky is gone, the fixes stop.
 */
class LocationManagerGnssSource(
    context: Context,
    private val sessionStartNs: Long,
    private val minTimeMs: Long = 1000L,
) : GnssSource {
    override val name = "LocationManager/gps"
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    @SuppressLint("MissingPermission") // the caller holds ACCESS_FINE_LOCATION (see @RequiresPermission)
    @RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
    override fun events(): Flow<GnssEvent> = callbackFlow {
        val thread = HandlerThread("sih26168-gnss").apply { start() }
        val listener = LocationListener { loc -> trySend(toEvent(loc, sessionStartNs)) }
        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, minTimeMs, 0f, listener, thread.looper)
        awaitClose {
            lm.removeUpdates(listener)
            thread.quitSafely()
        }
    }.buffer(capacity = 256)
}

/**
 * Google Play services fused provider (optional). It may blend Wi-Fi/cell/sensor positions:
 * better availability, but its fixes are NOT pure GNSS -- every logged line carries its
 * provider so a recorded drive says which it used.
 */
class FusedGnssSource(
    private val context: Context,
    private val sessionStartNs: Long,
    private val intervalMs: Long = 1000L,
) : GnssSource {
    override val name = "FusedLocationProvider"

    @SuppressLint("MissingPermission") // the caller holds ACCESS_FINE_LOCATION (see @RequiresPermission)
    @RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
    override fun events(): Flow<GnssEvent> = callbackFlow {
        val client = LocationServices.getFusedLocationProviderClient(context)
        val thread = HandlerThread("sih26168-fused").apply { start() }
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, intervalMs)
            .setMinUpdateIntervalMillis(intervalMs)
            .build()
        val callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                for (loc in result.locations) trySend(toEvent(loc, sessionStartNs))
            }
        }
        client.requestLocationUpdates(request, callback, thread.looper)
        awaitClose {
            client.removeLocationUpdates(callback)
            thread.quitSafely()
        }
    }.buffer(capacity = 256)
}
