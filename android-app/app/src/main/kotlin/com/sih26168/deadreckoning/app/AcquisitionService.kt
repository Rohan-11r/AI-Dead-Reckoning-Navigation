package com.sih26168.deadreckoning.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.sih26168.deadreckoning.core.Channel
import com.sih26168.deadreckoning.core.ChannelSample
import com.sih26168.deadreckoning.core.GnssOnlyEngine
import com.sih26168.deadreckoning.core.ImuAssembler
import com.sih26168.deadreckoning.core.NavigationEngine
import com.sih26168.deadreckoning.core.RateMeter
import com.sih26168.deadreckoning.core.SessionLogger
import com.sih26168.deadreckoning.core.SimulatedOutage
import com.sih26168.deadreckoning.core.TrackProjector
import com.sih26168.deadreckoning.core.Vec3
import com.sih26168.deadreckoning.core.nsToS
import com.sih26168.deadreckoning.gnss.GnssEvent
import com.sih26168.deadreckoning.gnss.LocationManagerGnssSource
import com.sih26168.deadreckoning.sensors.SensorSource
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.channels.Channel as KChannel

/**
 * Foreground service (type "location") that owns acquisition for the whole session.
 *
 * Coroutine layout (all on Dispatchers.Default -- never the main thread):
 *   sensors  : SensorSource.samples()        -> inputs   (callbacks on their own HandlerThread)
 *   gnss     : LocationManagerGnssSource      -> inputs   (callbacks on their own HandlerThread)
 *   outage   : repository.outageRequested    -> inputs   (UI toggle, as a command)
 *   probe    : OnnxLatencyProbe, once
 *   process  : the ONLY consumer of `inputs`; it alone touches the engine, the logger, the
 *              assembler, the outage switch and the rate meters -> no locks, ordered events.
 * The UI never sees raw rates: the processor publishes conflated StateFlow snapshots at
 * most every [PUBLISH_EVERY_S] of session time, so sensor arrival is decoupled from drawing.
 *
 * Session clock: SystemClock.elapsedRealtimeNanos() at start = t 0 for everything logged.
 */
class AcquisitionService : Service() {
    private sealed interface Input {
        data class Sensor(val s: ChannelSample) : Input
        data class Gnss(val e: GnssEvent) : Input
        data class Outage(val on: Boolean) : Input
    }

    private var scope: CoroutineScope? = null
    private val repo get() = (application as NavApplication).repository

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (scope == null) start()
        // a recording is never restarted behind the user's back after the process dies
        return START_NOT_STICKY
    }

    private fun start() {
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(),
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION else 0)
        val startNs = SystemClock.elapsedRealtimeNanos()
        val startUtcMs = System.currentTimeMillis()
        val stamp = SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }
            .format(Date(startUtcMs))
        val dir = File(getExternalFilesDir(null) ?: filesDir, "sessions/$stamp")
        val sensors = SensorSource(this, startNs)
        val gnss = LocationManagerGnssSource(this, startNs)
        val logger = SessionLogger(dir, mapOf(
            "session_start_utc_ms" to startUtcMs, "session_start_elapsed_ns" to startNs,
            "device_manufacturer" to Build.MANUFACTURER, "device_model" to Build.MODEL,
            "android_sdk" to Build.VERSION.SDK_INT, "app_version" to BuildConfigInfo.VERSION,
            "gnss_source" to gnss.name, "sensors_available" to sensors.available,
            "engine" to "gnss-only placeholder (DR engine not yet on device)",
        ))
        val s = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        scope = s
        repo.setRunning(true)
        val inputs = KChannel<Input>(capacity = 8192)

        s.launch { sensors.samples().collect { inputs.send(Input.Sensor(it)) } }
        s.launch {
            try {
                gnss.events().collect { inputs.send(Input.Gnss(it)) }
            } catch (x: SecurityException) {
                repo.reportError("location permission missing: GNSS not recorded (${x.message})")
            }
        }
        s.launch { repo.outageRequested.collect { inputs.send(Input.Outage(it)) } }
        val latency = AtomicReference<AiLatency?>(null) // written by the probe, read by process
        s.launch {
            latency.set(
                try {
                    OnnxLatencyProbe(assets).measure()
                } catch (x: Exception) {
                    AiLatency("error", null, null, 0, "probe failed: ${x.message}")
                },
            )
        }
        s.launch { process(inputs, startNs, dir, logger, sensors) { latency.get() } }
    }

    private suspend fun process(
        inputs: KChannel<Input>, startNs: Long, dir: File, logger: SessionLogger, sensors: SensorSource,
        latency: () -> AiLatency?,
    ) {
        val engine: NavigationEngine = GnssOnlyEngine()
        val sm = (engine as GnssOnlyEngine).sm
        val assembler = ImuAssembler()
        val outage = SimulatedOutage()
        val rates = Channel.entries.associateWith { RateMeter() }
        val gnssRate = RateMeter(windowS = 10.0, gapS = 30.0)
        val latest = HashMap<Channel, Vec3>()
        val track = TrackProjector()
        var nTransitions = 0
        var fixesOk = 0L
        var fixesBad = 0L
        var lastFixRejection: String? = null
        var lastPublish = Double.NEGATIVE_INFINITY
        var now = 0.0
        try {
            for (input in inputs) {
                when (input) {
                    is Input.Sensor -> {
                        val c = input.s
                        now = maxOf(now, c.tS)
                        logger.sensor(c)
                        rates.getValue(c.channel).record(c.tS)
                        latest[c.channel] = c.value
                        engine.onChannel(c)
                        assembler.offer(c)?.let { engine.onImu(it) }
                    }
                    is Input.Gnss -> when (val e = input.e) {
                        is GnssEvent.Fix -> {
                            fixesOk++
                            gnssRate.record(e.fix.tReceivedS)
                            val withheld = outage.withholds(e.fix)
                            logger.fix(e.fix, e.provider, withheld)
                            if (!withheld) {
                                engine.onFix(e.fix)
                                engine.snapshot().let { s -> if (s.latRad != null && s.lonRad != null) track.add(s.latRad!!, s.lonRad!!) }
                            }
                        }
                        is GnssEvent.Rejected -> {
                            fixesBad++
                            lastFixRejection = e.reason
                            logger.event(now, "fix_rejected", "${e.provider}: ${e.reason}")
                        }
                    }
                    is Input.Outage -> {
                        val t = nsToS(SystemClock.elapsedRealtimeNanos(), startNs)
                        if (input.on != outage.enabled) {
                            outage.set(input.on, t)
                            logger.event(t, if (input.on) "sim_outage_start" else "sim_outage_end")
                        }
                    }
                }
                while (nTransitions < sm.transitions.size) {
                    val tr = sm.transitions[nTransitions++]
                    logger.event(tr.tS, "gnss_state", "${tr.from} -> ${tr.to}: ${tr.why}")
                }
                if (now - lastPublish >= PUBLISH_EVERY_S) {
                    lastPublish = now
                    val st = sensors.stats
                    repo.publish(
                        engine.snapshot().copy(aiLatencyMs = latency()?.medianMs),
                        repo.diagnostics.value.copy(
                            ratesHz = rates.mapValues { it.value.rateHz }, latest = HashMap(latest),
                            gnssRateHz = gnssRate.rateHz, sensorsAvailable = sensors.available,
                            sensorMapped = st.mapped, sensorRejected = st.rejected, sensorDropped = st.dropped,
                            lastSensorRejection = st.lastRejection, imuWithoutGyro = assembler.nWithoutGyro,
                            fixesAccepted = fixesOk, fixesRejected = fixesBad, lastFixRejection = lastFixRejection,
                            fixesWithheldBySim = outage.nWithheld, loggerSensorLines = logger.nSensorLines,
                            loggerGnssLines = logger.nGnssLines, sessionDir = dir.absolutePath,
                            covarianceDiag = engine.snapshot().covarianceDiag, aiLatency = latency(),
                        ),
                        track.points,
                    )
                }
            }
        } finally {
            logger.close() // runs on cancellation too: the recording is always closed cleanly
        }
    }

    override fun onDestroy() {
        scope?.cancel()
        scope = null
        repo.setRunning(false)
        super.onDestroy()
    }

    private fun notification(): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, getString(R.string.notification_channel),
            NotificationManager.IMPORTANCE_LOW))
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.notification_text))
            .setOngoing(true)
            .build()
    }

    companion object {
        const val ACTION_STOP = "com.sih26168.deadreckoning.STOP"
        private const val CHANNEL_ID = "acquisition"
        private const val NOTIFICATION_ID = 26168
        private const val PUBLISH_EVERY_S = 0.1
    }
}

/** Version string without enabling the BuildConfig feature. */
internal object BuildConfigInfo {
    const val VERSION = "0.10.0-phase10"
}
