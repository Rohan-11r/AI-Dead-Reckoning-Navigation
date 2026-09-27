package com.sih26168.deadreckoning.core

import java.io.BufferedWriter
import java.io.File
import java.io.OutputStreamWriter
import java.util.Locale

/**
 * Records a drive to disk for dataset expansion -- the phone-side counterpart of the
 * IO-VNBD "S" files, in a documented schema (`SCHEMA_VERSION`), as plain CSV + a JSON
 * manifest in one directory per session:
 *
 *   manifest.json   schema, session start (UTC ms + elapsed-realtime ns), device, app, end
 *   sensors.csv     t_s,channel,x,y,z,accuracy          (every raw ChannelSample, DEVICE frame)
 *   gnss.csv        t_s,t_received_s,lat_deg,lon_deg,h_m,accuracy_m,speed_mps,bearing_deg,
 *                   sats_used,provider,withheld_by_sim
 *   events.csv      t_s,kind,detail                     (state transitions, outage toggles, notes)
 *
 * Numbers are written locale-independently with full round-trip precision (Double.toString),
 * so a German-locale phone writes 9.81, not 9,81. Buffered; flushed every [flushEveryLines]
 * lines and on [flush]/[close]. Methods are synchronized: the acquisition service may log
 * from its sensor and GNSS coroutines. The logger never drops a line silently: a write
 * after [close] throws.
 */
class SessionLogger(
    val dir: File,
    private val manifest: Map<String, Any?>,
    private val flushEveryLines: Int = 500,
) : AutoCloseable {
    companion object {
        const val SCHEMA_VERSION = 1
        val SENSORS_HEADER = listOf("t_s", "channel", "x", "y", "z", "accuracy")
        val GNSS_HEADER = listOf("t_s", "t_received_s", "lat_deg", "lon_deg", "h_m", "accuracy_m", "speed_mps",
            "bearing_deg", "sats_used", "provider", "withheld_by_sim")
        val EVENTS_HEADER = listOf("t_s", "kind", "detail")
    }

    private val sensors: BufferedWriter
    private val gnss: BufferedWriter
    private val events: BufferedWriter
    private var pending = 0
    var closed = false
        private set
    var nSensorLines = 0L
        private set
    var nGnssLines = 0L
        private set

    init {
        require(dir.isDirectory || dir.mkdirs()) { "cannot create session directory $dir" }
        sensors = open("sensors.csv", SENSORS_HEADER)
        gnss = open("gnss.csv", GNSS_HEADER)
        events = open("events.csv", EVENTS_HEADER)
        writeManifest(extra = mapOf("closed" to false))
    }

    private fun open(name: String, header: List<String>): BufferedWriter =
        BufferedWriter(OutputStreamWriter(File(dir, name).outputStream(), Charsets.UTF_8), 1 shl 16).also {
            it.write(header.joinToString(","))
            it.write("\n")
        }

    @Synchronized
    fun sensor(s: ChannelSample) {
        line(sensors, listOf(num(s.tS), s.channel.name, num(s.value.x), num(s.value.y), num(s.value.z), s.accuracy.toString()))
        nSensorLines++
    }

    @Synchronized
    fun fix(f: GnssSample, provider: String, withheldBySim: Boolean) {
        val deg = 180.0 / Math.PI
        line(gnss, listOf(num(f.tS), num(f.tReceivedS), num(f.latRad * deg), num(f.lonRad * deg), num(f.hM),
            num(f.horizontalAccuracyM), f.speedMps?.let(::num) ?: "", f.bearingRad?.let { num(it * deg) } ?: "",
            f.satsUsed?.toString() ?: "", csvText(provider), if (withheldBySim) "1" else "0"))
        nGnssLines++
    }

    @Synchronized
    fun event(tS: Double, kind: String, detail: String = "") {
        line(events, listOf(num(tS), csvText(kind), csvText(detail)))
    }

    @Synchronized
    fun flush() {
        check(!closed) { "logger closed" }
        sensors.flush(); gnss.flush(); events.flush()
        pending = 0
    }

    @Synchronized
    override fun close() {
        if (closed) return
        flush()
        sensors.close(); gnss.close(); events.close()
        closed = true
        writeManifest(extra = mapOf("closed" to true, "sensor_lines" to nSensorLines, "gnss_lines" to nGnssLines))
    }

    private fun line(w: BufferedWriter, cells: List<String>) {
        check(!closed) { "logger closed: a line would be lost" }
        w.write(cells.joinToString(","))
        w.write("\n")
        if (++pending >= flushEveryLines) flush()
    }

    private fun writeManifest(extra: Map<String, Any?>) {
        val all = linkedMapOf<String, Any?>("schema_version" to SCHEMA_VERSION, "frame" to "DEVICE (Android sensor axes)",
            "time_base" to "t_s = (elapsedRealtimeNanos - session_start_elapsed_ns) / 1e9") + manifest + extra
        File(dir, "manifest.json").writeText(toJson(all) + "\n", Charsets.UTF_8)
    }
}

/** Round-trip-exact, locale-independent number text. */
internal fun num(x: Double): String = if (x.isFinite()) x.toString() else ""

/** Quote a CSV text cell when needed (RFC 4180). */
internal fun csvText(s: String): String =
    if (s.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) "\"" + s.replace("\"", "\"\"") + "\"" else s

/** Minimal JSON writer for flat/nested maps of strings, numbers, booleans, null and lists. */
internal fun toJson(v: Any?): String = when (v) {
    null -> "null"
    is String -> "\"" + v.flatMap { c ->
        when (c) {
            '"' -> listOf('\\', '"'); '\\' -> listOf('\\', '\\'); '\n' -> listOf('\\', 'n')
            '\r' -> listOf('\\', 'r'); '\t' -> listOf('\\', 't')
            else -> if (c < ' ') String.format(Locale.ROOT, "\\u%04x", c.code).toList() else listOf(c)
        }
    }.joinToString("") + "\""
    is Boolean -> v.toString()
    is Double -> if (v.isFinite()) v.toString() else "null"
    is Float -> if (v.isFinite()) v.toString() else "null"
    is Number -> v.toString()
    is Map<*, *> -> v.entries.joinToString(",", "{", "}") { (k, x) -> toJson(k.toString()) + ":" + toJson(x) }
    is Iterable<*> -> v.joinToString(",", "[", "]") { toJson(it) }
    else -> toJson(v.toString())
}
