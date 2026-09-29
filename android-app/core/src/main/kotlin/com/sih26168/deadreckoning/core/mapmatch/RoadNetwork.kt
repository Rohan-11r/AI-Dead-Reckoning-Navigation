package com.sih26168.deadreckoning.core.mapmatch

import com.sih26168.deadreckoning.core.nav.LocalTangentPlane
import java.io.DataInputStream
import java.io.InputStream
import java.util.PriorityQueue
import kotlin.math.atan2
import kotlin.math.floor
import kotlin.math.hypot

/**
 * Port of navcore.map_matching.road_network, read from the ROAD BUNDLE written by
 * navcore.map_matching.bundle (big-endian: DataInputStream order). Same directed segments,
 * same ids, same 50 m grid rule, same bounded Dijkstra, same tangent plane.
 * OFFLINE: reads a stream the app opens from its assets/files; no network code.
 * Parity: nav_map_matching.json (+ mm_*.roads.bin).
 */
data class Candidate(val seg: Int, val x: Double, val y: Double, val frac: Double, val distM: Double)

class RoadNetwork(
    val ltp: LocalTangentPlane,
    val nodeX: DoubleArray,
    val nodeY: DoubleArray,
    val segFrom: IntArray,
    val segTo: IntArray,
    val segWay: LongArray,
    val wayTags: Map<Long, Map<String, String>>,
    val cellM: Double,
) {
    val nSegments = segFrom.size
    val segAx = DoubleArray(nSegments) { nodeX[segFrom[it]] }
    val segAy = DoubleArray(nSegments) { nodeY[segFrom[it]] }
    val segDx = DoubleArray(nSegments) { nodeX[segTo[it]] - segAx[it] }
    val segDy = DoubleArray(nSegments) { nodeY[segTo[it]] - segAy[it] }
    val segLen = DoubleArray(nSegments) { hypot(segDx[it], segDy[it]) }
    val segHeading = DoubleArray(nSegments) { atan2(segDy[it], segDx[it]) }
    private val outSegs: Map<Int, IntArray>
    private val grid: Map<Long, IntArray>

    init {
        val out = HashMap<Int, MutableList<Int>>()
        for (s in 0 until nSegments) out.getOrPut(segFrom[s]) { mutableListOf() } += s  // ascending ids
        outSegs = out.mapValues { it.value.toIntArray() }
        val g = HashMap<Long, MutableList<Int>>()
        for (s in 0 until nSegments) {
            val bx = segAx[s] + segDx[s]
            val by = segAy[s] + segDy[s]
            val i0 = floor(minOf(segAx[s], bx) / cellM).toLong()
            val i1 = floor(maxOf(segAx[s], bx) / cellM).toLong()
            val j0 = floor(minOf(segAy[s], by) / cellM).toLong()
            val j1 = floor(maxOf(segAy[s], by) / cellM).toLong()
            for (i in i0..i1) for (j in j0..j1) g.getOrPut(key(i, j)) { mutableListOf() } += s
        }
        grid = g.mapValues { it.value.toIntArray() }
    }

    private fun key(i: Long, j: Long): Long = (i shl 32) xor (j and 0xffffffffL)

    fun segmentsNear(x: Double, y: Double, radiusM: Double): IntArray {
        val i0 = floor((x - radiusM) / cellM).toLong()
        val i1 = floor((x + radiusM) / cellM).toLong()
        val j0 = floor((y - radiusM) / cellM).toLong()
        val j1 = floor((y + radiusM) / cellM).toLong()
        val set = java.util.TreeSet<Int>()  // np.unique: sorted, unique
        for (i in i0..i1) for (j in j0..j1) grid[key(i, j)]?.forEach { set += it }
        return set.toIntArray()
    }

    /** Closest point on segment [s] to (x, y): (px, py, frac, dist). */
    fun project(s: Int, x: Double, y: Double): DoubleArray {
        val l2 = maxOf(segDx[s] * segDx[s] + segDy[s] * segDy[s], 1e-12)
        val frac = (((x - segAx[s]) * segDx[s] + (y - segAy[s]) * segDy[s]) / l2).coerceIn(0.0, 1.0)
        val px = segAx[s] + frac * segDx[s]
        val py = segAy[s] + frac * segDy[s]
        return doubleArrayOf(px, py, frac, hypot(px - x, py - y))
    }

    /** Every directed segment within [radiusM], ordered by segment id. */
    fun candidates(x: Double, y: Double, radiusM: Double): List<Candidate> =
        segmentsNear(x, y, radiusM).map { s ->
            val p = project(s, x, y)
            if (p[3] <= radiusM) Candidate(s, p[0], p[1], p[2], p[3]) else null
        }.filterNotNull()

    /** Shortest directed distances from [node] to every node within [cutoffM]. */
    fun distancesFromNode(node: Int, cutoffM: Double): Map<Int, Double> {
        val dist = HashMap<Int, Double>()
        dist[node] = 0.0
        val heap = PriorityQueue<Pair<Double, Int>>(compareBy<Pair<Double, Int>> { it.first }.thenBy { it.second })
        heap += 0.0 to node
        while (heap.isNotEmpty()) {
            val (d, n) = heap.poll()
            if (d > (dist[n] ?: Double.POSITIVE_INFINITY)) continue
            for (s in outSegs[n] ?: continue) {
                val nd = d + segLen[s]
                val m = segTo[s]
                if (nd <= cutoffM && nd < (dist[m] ?: Double.POSITIVE_INFINITY)) {
                    dist[m] = nd
                    heap += nd to m
                }
            }
        }
        return dist
    }

    /** Driving distance from [a] to [b] respecting one-ways; +inf if no route within [cutoffM]. */
    fun routeDistance(a: Candidate, b: Candidate, cutoffM: Double, cache: MutableMap<Int, Map<Int, Double>>? = null): Double {
        if (a.seg == b.seg && b.frac >= a.frac) return (b.frac - a.frac) * segLen[a.seg]
        val rest = (1.0 - a.frac) * segLen[a.seg]
        val into = b.frac * segLen[b.seg]
        if (rest + into > cutoffM) return Double.POSITIVE_INFINITY
        val src = segTo[a.seg]
        val table = if (cache != null) cache.getOrPut(src) { distancesFromNode(src, cutoffM) } else distancesFromNode(src, cutoffM)
        val mid = table[segFrom[b.seg]] ?: Double.POSITIVE_INFINITY
        val total = rest + mid + into
        return if (total <= cutoffM) total else Double.POSITIVE_INFINITY
    }

    fun toXy(latRad: Double, lonRad: Double): DoubleArray = ltp.geodeticToEnu(latRad, lonRad, 0.0)

    fun toGeodetic(x: Double, y: Double, u: Double = 0.0): DoubleArray = ltp.enuToGeodetic(doubleArrayOf(x, y, u))

    fun wayOf(seg: Int): Map<String, String> = wayTags[segWay[seg]] ?: emptyMap()

    fun roadName(seg: Int): String? = wayOf(seg)["name"]

    companion object {
        const val MAGIC = "SIHROAD1"
        const val VERSION = 1

        /** Read a bundle; throws on a wrong magic/version or trailing bytes. */
        fun read(input: InputStream): RoadNetwork = DataInputStream(input.buffered()).use { d ->
            val magic = ByteArray(8).also { d.readFully(it) }.toString(Charsets.US_ASCII)
            require(magic == MAGIC) { "not a road bundle ($magic)" }
            val ver = d.readInt()
            require(ver == VERSION) { "bundle version $ver, expected $VERSION" }
            val lat0 = d.readDouble()
            val lon0 = d.readDouble()
            val h0 = d.readDouble()
            val cell = d.readDouble()
            val n = d.readInt()
            val xs = DoubleArray(n)
            val ys = DoubleArray(n)
            for (i in 0 until n) { xs[i] = d.readDouble(); ys[i] = d.readDouble() }
            val m = d.readInt()
            val sf = IntArray(m) { d.readInt() }
            val st = IntArray(m) { d.readInt() }
            val sw = LongArray(m) { d.readLong() }
            val k = d.readInt()
            val tagsJson = ByteArray(k).also { d.readFully(it) }.toString(Charsets.UTF_8)
            require(d.read() == -1) { "trailing bytes after the road bundle" }
            RoadNetwork(LocalTangentPlane(lat0, lon0, h0), xs, ys, sf, st, sw, parseTags(tagsJson), cell)
        }

        /** {"way_id": {"k": "v"}} -- flat string maps only, so a tiny parser suffices. */
        internal fun parseTags(json: String): Map<Long, Map<String, String>> {
            val o = org.json.JSONObject(json)
            val out = HashMap<Long, Map<String, String>>()
            for (w in o.keys()) {
                val t = o.getJSONObject(w)
                out[w.toLong()] = t.keys().asSequence().associateWith { t.getString(it) }
            }
            return out
        }
    }
}
