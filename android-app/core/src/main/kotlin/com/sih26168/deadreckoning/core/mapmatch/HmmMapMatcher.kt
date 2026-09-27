package com.sih26168.deadreckoning.core.mapmatch

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Port of navcore.map_matching.hmm (online HMM, Newson-Krumm) and navcore.map_matching.outage
 * (the dead-reckoning wrapper). REFINEMENT ONLY: never writes to the filter; returns no
 * position when no road plausibly explains the observation. Emission: Mahalanobis with the
 * filter's 2x2 covariance + map error, plus heading agreement; transition -|route - travelled|/beta,
 * probability 0 without a route; forward filter output + Viterbi decode; off-network
 * suspension. Parity: nav_map_matching.json.
 */
data class MatcherConfig(
    val kSigma: Double = 3.0,
    val rMinM: Double = 30.0,
    val rMaxM: Double = 300.0,
    val maxCandidates: Int = 12,
    val sigmaMapM: Double = 5.0,
    val betaM: Double = 10.0,
    val routeCutoffFactor: Double = 2.0,
    val routeCutoffSlackM: Double = 100.0,
    val headingMinSpeedMps: Double = 2.0,
    val sigmaHeadingRoadRad: Double = Math.toRadians(20.0),
    val chi2Gate: Double = 13.8,
    val suspendAfter: Int = 3,
    val reenterAfter: Int = 3,
    val agreeRadiusM: Double = 10.0,
    val viterbiMaxEpochs: Int = 3600,
)

data class MapMatchResult(
    val tS: Double,
    val matched: Boolean,
    val mode: String,                 // matched | implausible | suspended | no_candidates
    val xM: Double? = null,
    val yM: Double? = null,
    val latRad: Double? = null,
    val lonRad: Double? = null,
    val confidence: Double = 0.0,     // map_match_confidence
    val segment: Int? = null,
    val roadName: String? = null,
    val correctionM: Double? = null,
    val nCandidates: Int = 0,
    val searchRadiusM: Double = 0.0,
)

private class Epoch(val tS: Double, val cands: List<Candidate>, val delta: DoubleArray, val back: IntArray?)

private fun wrap(a: Double): Double = (a + PI).mod(2 * PI) - PI

/** log(sum(exp(v))) with the reference's guard: an all -inf vector gives -inf. */
private fun logSumExp(v: DoubleArray): Double {
    var m = Double.NEGATIVE_INFINITY
    for (x in v) if (x > m) m = x
    val mm = if (m.isFinite()) m else 0.0
    var s = 0.0
    for (x in v) s += exp(x - mm)
    return mm + ln(s)
}

private fun argmax(v: DoubleArray): Int {
    var k = 0
    for (i in 1 until v.size) if (v[i] > v[k]) k = i
    return k
}

class HmmMapMatcher(val net: RoadNetwork, val cfg: MatcherConfig = MatcherConfig()) {
    val counts = linkedMapOf<String, Int>()
    private var prev: List<Candidate>? = null
    private var alpha: DoubleArray? = null
    private var prevXy: DoubleArray? = null
    private val epochs = ArrayDeque<Epoch>()
    private var bad = 0
    private var good = 0
    private var travel = 0.0
    var suspended = false
        private set

    fun reset() {
        prev = null; alpha = null; prevXy = null; epochs.clear(); bad = 0; good = 0; travel = 0.0; suspended = false
    }

    private fun count(k: String) {
        counts[k] = (counts[k] ?: 0) + 1
    }

    /** (x, y) in the network plane [m]; pPos = 2x2 EN covariance [a, b, b, d] (row-major). */
    fun step(
        tS: Double, x: Double, y: Double, pPos: DoubleArray, headingRad: Double? = null, speedMps: Double = 0.0,
        sigmaHeadingRad: Double = Math.toRadians(10.0), travelledM: Double? = null, u: Double = 0.0,
    ): MapMatchResult {
        val s2 = cfg.sigmaMapM * cfg.sigmaMapM
        val a = pPos[0] + s2
        val b = pPos[1]
        val c2 = pPos[2]
        val d = pPos[3] + s2
        // largest eigenvalue of the symmetric 2x2 S (the reference uses S[0,1]; S is symmetric)
        val lmax = 0.5 * (a + d) + sqrt(0.25 * (a - d) * (a - d) + b * b)
        val r = (cfg.kSigma * sqrt(lmax)).coerceIn(cfg.rMinM, cfg.rMaxM)
        val pxy = prevXy
        val tr = travelledM ?: (if (pxy == null) 0.0 else hypot(x - pxy[0], y - pxy[1]))
        prevXy = doubleArrayOf(x, y)
        travel += tr
        var cands = net.candidates(x, y, r)
        val heading = if (headingRad != null && speedMps >= cfg.headingMinSpeedMps) headingRad else null
        var loge = DoubleArray(0)
        val plausible: Boolean
        if (cands.isNotEmpty()) {
            val det = a * d - b * c2
            val i00 = d / det
            val i01 = -b / det
            val i10 = -c2 / det
            val i11 = a / det
            val m2All = DoubleArray(cands.size) { k ->
                val dx = cands[k].x - x
                val dy = cands[k].y - y
                dx * (i00 * dx + i01 * dy) + dy * (i10 * dx + i11 * dy)
            }
            val logeAll = DoubleArray(cands.size) { -0.5 * m2All[it] }
            if (heading != null) {
                val kappa = 1.0 / (sigmaHeadingRad * sigmaHeadingRad + cfg.sigmaHeadingRoadRad * cfg.sigmaHeadingRoadRad)
                for (k in cands.indices) logeAll[k] += kappa * (cos(wrap(net.segHeading[cands[k].seg] - heading)) - 1.0)
            }
            val keep = cands.indices.sortedWith(compareBy<Int> { m2All[it] }.thenBy { it }).take(cfg.maxCandidates).sorted()
            cands = keep.map { cands[it] }
            loge = DoubleArray(keep.size) { logeAll[keep[it]] }
            plausible = keep.minOf { m2All[it] } <= cfg.chi2Gate
        } else {
            plausible = false
        }

        if (plausible) { good++; bad = 0 } else { bad++; good = 0 }
        if (!suspended && bad >= cfg.suspendAfter) {
            suspended = true
            count("suspensions")
            prev = null; alpha = null; travel = 0.0
        }
        if (suspended) {
            if (good >= cfg.reenterAfter) {
                suspended = false
                count("reentries")
            } else {
                count("epochs_suspended")
                return MapMatchResult(tS, false, "suspended", nCandidates = cands.size, searchRadiusM = r)
            }
        }
        if (!plausible) {
            val mode = if (cands.isNotEmpty()) "implausible" else "no_candidates"
            count("epochs_$mode")
            return MapMatchResult(tS, false, mode, nCandidates = cands.size, searchRadiusM = r)
        }

        var back: IntArray? = null
        var al: DoubleArray
        var de: DoubleArray
        val pv = prev
        if (pv == null) {
            al = loge.copyOf(); de = loge.copyOf()
        } else {
            val T = transition(pv, cands, travel)
            if (T.all { row -> row.all { !it.isFinite() } }) {
                count("breaks")
                al = loge.copyOf(); de = loge.copyOf()
            } else {
                val prevAlpha = alpha!!
                val prevDelta = epochs.last().delta
                al = DoubleArray(cands.size) { j -> logSumExp(DoubleArray(pv.size) { i -> prevAlpha[i] + T[i][j] }) + loge[j] }
                val bk = IntArray(cands.size) { j -> argmax(DoubleArray(pv.size) { i -> prevDelta[i] + T[i][j] }) }
                de = DoubleArray(cands.size) { j -> prevDelta[bk[j]] + T[bk[j]][j] + loge[j] }
                back = bk
                if (al.none { it.isFinite() }) {
                    count("breaks")
                    al = loge.copyOf(); de = loge.copyOf(); back = null
                }
            }
        }
        val lse = logSumExp(al)
        al = DoubleArray(al.size) { al[it] - lse }
        val dmax = de.max()
        de = DoubleArray(de.size) { de[it] - dmax }
        if (back == null) {
            epochs.clear()  // a break starts a new decodable chain
            epochs.addLast(Epoch(tS, cands, de, null))
        } else {
            epochs.addLast(Epoch(tS, cands, de, back))
            while (epochs.size > cfg.viterbiMaxEpochs) epochs.removeFirst()
        }
        prev = cands; alpha = al; travel = 0.0

        val post = DoubleArray(al.size) { exp(al[it]) }
        val j = argmax(post)
        val c = cands[j]
        var conf = 0.0
        for (k in cands.indices) if (hypot(cands[k].x - c.x, cands[k].y - c.y) <= cfg.agreeRadiusM) conf += post[k]
        val g = net.toGeodetic(c.x, c.y, u)
        count("epochs_matched")
        return MapMatchResult(tS, true, "matched", c.x, c.y, g[0], g[1], minOf(1.0, conf), c.seg, net.roadName(c.seg),
            hypot(c.x - x, c.y - y), cands.size, r)
    }

    private fun transition(prev: List<Candidate>, cands: List<Candidate>, travelled: Double): Array<DoubleArray> {
        val cutoff = cfg.routeCutoffFactor * travelled + cfg.routeCutoffSlackM
        val cache = HashMap<Int, Map<Int, Double>>()
        return Array(prev.size) { i ->
            DoubleArray(cands.size) { j ->
                val r = net.routeDistance(prev[i], cands[j], cutoff, cache)
                if (r.isFinite()) -abs(r - travelled) / cfg.betaM else Double.NEGATIVE_INFINITY
            }
        }
    }

    /** Viterbi path over the current chain (since the last break): (t, candidate). */
    fun decode(): List<Pair<Double, Candidate>> {
        if (epochs.isEmpty()) return emptyList()
        var j = argmax(epochs.last().delta)
        val path = mutableListOf<Pair<Double, Candidate>>()
        for (ep in epochs.reversed()) {
            path += ep.tS to ep.cands[j]
            val b = ep.back ?: break
            j = b[j]
        }
        return path.reversed()
    }
}

/** Travel direction atan2(vN, vE), its 1-sigma (first order), and the horizontal speed. */
fun courseAndSigma(vEn: DoubleArray, pV: DoubleArray): DoubleArray {
    val ve = vEn[0]
    val vn = vEn[1]
    val s2 = ve * ve + vn * vn
    val speed = sqrt(s2)
    if (speed < 1e-3) return doubleArrayOf(0.0, PI, speed)
    val j0 = -vn / s2
    val j1 = ve / s2
    val v = j0 * (pV[0] * j0 + pV[1] * j1) + j1 * (pV[2] * j0 + pV[3] * j1)
    return doubleArrayOf(atan2(vn, ve), sqrt(maxOf(v, 0.0)), speed)
}

/**
 * Port of navcore.map_matching.outage.DeadReckoningMapMatcher: feed every fused sample;
 * returns a road-constrained position at most every [periodS], accumulating the path
 * travelled between epochs. Reads the filter, never writes it.
 */
class DeadReckoningMapMatcher(val net: RoadNetwork, cfg: MatcherConfig = MatcherConfig(), rateHz: Double = 1.0) {
    val hmm = HmmMapMatcher(net, cfg)
    private val periodS = 1.0 / rateHz
    private var lastT = Double.NEGATIVE_INFINITY
    private var lastXy: DoubleArray? = null
    private var pathM = 0.0
    var last: MapMatchResult? = null
        private set

    /** pPos, pV: 2x2 row-major covariances of the EN position and velocity. */
    fun update(tS: Double, latRad: Double, lonRad: Double, pPos: DoubleArray, vEn: DoubleArray, pV: DoubleArray): MapMatchResult? {
        val xyz = net.toXy(latRad, lonRad)
        val lx = lastXy
        if (lx != null) pathM += hypot(xyz[0] - lx[0], xyz[1] - lx[1])
        lastXy = doubleArrayOf(xyz[0], xyz[1])
        if (tS - lastT < periodS - 1e-9) return null
        val cs = courseAndSigma(vEn, pV)
        val travelled = if (lastT.isFinite()) pathM else null
        lastT = tS
        pathM = 0.0
        return hmm.step(tS, xyz[0], xyz[1], pPos, cs[0], cs[2], cs[1], travelled, xyz[2]).also { last = it }
    }
}
