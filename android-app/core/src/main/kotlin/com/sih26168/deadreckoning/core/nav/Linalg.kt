package com.sih26168.deadreckoning.core.nav

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Small dense linear algebra for the 15-state filter: row-major DoubleArray matrices with
 * explicit dimensions. Plain loops, double throughout (AGENTS.md 4: accumulators are
 * double). Summation order differs from numpy/BLAS, so results differ from the Python
 * reference at the ~1e-16 relative level per operation; the golden-vector tests bound the
 * accumulated difference.
 */
class Mat(val rows: Int, val cols: Int, val a: DoubleArray = DoubleArray(rows * cols)) {
    init {
        require(a.size == rows * cols) { "Mat ${rows}x$cols needs ${rows * cols} values, got ${a.size}" }
    }

    operator fun get(i: Int, j: Int): Double = a[i * cols + j]
    operator fun set(i: Int, j: Int, v: Double) {
        a[i * cols + j] = v
    }

    fun copy(): Mat = Mat(rows, cols, a.copyOf())

    operator fun times(o: Mat): Mat {
        require(cols == o.rows) { "shape ${rows}x$cols * ${o.rows}x${o.cols}" }
        val r = Mat(rows, o.cols)
        for (i in 0 until rows) {
            for (k in 0 until cols) {
                val x = a[i * cols + k]
                if (x == 0.0) continue
                for (j in 0 until o.cols) r.a[i * o.cols + j] += x * o.a[k * o.cols + j]
            }
        }
        return r
    }

    operator fun plus(o: Mat): Mat {
        require(rows == o.rows && cols == o.cols)
        return Mat(rows, cols, DoubleArray(a.size) { a[it] + o.a[it] })
    }

    operator fun minus(o: Mat): Mat {
        require(rows == o.rows && cols == o.cols)
        return Mat(rows, cols, DoubleArray(a.size) { a[it] - o.a[it] })
    }

    operator fun times(s: Double): Mat = Mat(rows, cols, DoubleArray(a.size) { a[it] * s })

    fun t(): Mat {
        val r = Mat(cols, rows)
        for (i in 0 until rows) for (j in 0 until cols) r.a[j * rows + i] = a[i * cols + j]
        return r
    }

    fun mulVec(v: DoubleArray): DoubleArray {
        require(v.size == cols)
        return DoubleArray(rows) { i -> var s = 0.0; for (k in 0 until cols) s += a[i * cols + k] * v[k]; s }
    }

    fun diag(): DoubleArray = DoubleArray(minOf(rows, cols)) { this[it, it] }

    /** Copy [src] into this matrix at (r0, c0). */
    fun setBlock(r0: Int, c0: Int, src: Mat) {
        for (i in 0 until src.rows) for (j in 0 until src.cols) this[r0 + i, c0 + j] = src[i, j]
    }

    fun block(r0: Int, c0: Int, nr: Int, nc: Int): Mat {
        val r = Mat(nr, nc)
        for (i in 0 until nr) for (j in 0 until nc) r[i, j] = this[r0 + i, c0 + j]
        return r
    }

    fun isFinite(): Boolean = a.all { it.isFinite() }

    companion object {
        fun eye(n: Int): Mat = Mat(n, n).also { for (i in 0 until n) it[i, i] = 1.0 }
        fun diag(v: DoubleArray): Mat = Mat(v.size, v.size).also { for (i in v.indices) it[i, i] = v[i] }
        fun of(rows: Int, cols: Int, vararg v: Double): Mat = Mat(rows, cols, v.copyOf())
    }
}

class NotPositiveDefinite(message: String) : ArithmeticException(message)

/** Cholesky factor L (lower) of a symmetric positive-definite matrix, or throw. */
fun cholesky(m: Mat): Mat {
    require(m.rows == m.cols)
    val n = m.rows
    val l = Mat(n, n)
    for (j in 0 until n) {
        var d = m[j, j]
        for (k in 0 until j) d -= l[j, k] * l[j, k]
        if (!(d > 0.0)) throw NotPositiveDefinite("leading minor $j is $d")
        val ljj = sqrt(d)
        l[j, j] = ljj
        for (i in j + 1 until n) {
            var s = m[i, j]
            for (k in 0 until j) s -= l[i, k] * l[j, k]
            l[i, j] = s / ljj
        }
    }
    return l
}

/** Solve A X = B (A square) by Gaussian elimination with partial pivoting, like LAPACK gesv. */
fun solve(A: Mat, B: Mat): Mat {
    require(A.rows == A.cols && A.rows == B.rows)
    val n = A.rows
    val m = B.cols
    val a = A.copy()
    val b = B.copy()
    for (c in 0 until n) {
        var p = c
        var best = abs(a[c, c])
        for (r in c + 1 until n) if (abs(a[r, c]) > best) {
            best = abs(a[r, c]); p = r
        }
        if (best == 0.0) throw ArithmeticException("singular matrix")
        if (p != c) {
            for (j in 0 until n) { val t = a[c, j]; a[c, j] = a[p, j]; a[p, j] = t }
            for (j in 0 until m) { val t = b[c, j]; b[c, j] = b[p, j]; b[p, j] = t }
        }
        for (r in c + 1 until n) {
            val f = a[r, c] / a[c, c]
            if (f == 0.0) continue
            for (j in c until n) a[r, j] -= f * a[c, j]
            for (j in 0 until m) b[r, j] -= f * b[c, j]
        }
    }
    val x = Mat(n, m)
    for (j in 0 until m) {
        for (i in n - 1 downTo 0) {
            var s = b[i, j]
            for (k in i + 1 until n) s -= a[i, k] * x[k, j]
            x[i, j] = s / a[i, i]
        }
    }
    return x
}

fun solveVec(A: Mat, v: DoubleArray): DoubleArray = solve(A, Mat(v.size, 1, v.copyOf())).a

// ---- 3-vectors as DoubleArray(3) --------------------------------------------------------

fun v3(x: Double, y: Double, z: Double) = doubleArrayOf(x, y, z)
fun cross(a: DoubleArray, b: DoubleArray) =
    doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
fun dot(a: DoubleArray, b: DoubleArray): Double { var s = 0.0; for (i in a.indices) s += a[i] * b[i]; return s }
fun norm(a: DoubleArray): Double = sqrt(dot(a, a))
fun add(a: DoubleArray, b: DoubleArray) = DoubleArray(a.size) { a[it] + b[it] }
fun sub(a: DoubleArray, b: DoubleArray) = DoubleArray(a.size) { a[it] - b[it] }
fun scale(a: DoubleArray, s: Double) = DoubleArray(a.size) { a[it] * s }

/** [a]x such that skew(a) * u == cross(a, u). */
fun skew(a: DoubleArray): Mat = Mat.of(3, 3, 0.0, -a[2], a[1], a[2], 0.0, -a[0], -a[1], a[0], 0.0)

fun rotZ(angle: Double): Mat {
    val c = kotlin.math.cos(angle)
    val s = kotlin.math.sin(angle)
    return Mat.of(3, 3, c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}
