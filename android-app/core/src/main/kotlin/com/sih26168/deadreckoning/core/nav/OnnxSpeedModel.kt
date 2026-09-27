package com.sih26168.deadreckoning.core.nav

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.FloatBuffer
import java.security.MessageDigest

/**
 * Model A (forward speed) through ONNX Runtime: the port of
 * navcore.fusion.ai_models.OnnxSequenceModel. The model and its card are read as BYTES
 * (the app passes APK assets, the tests pass files), and the model is REFUSED unless
 *  - its SHA-256 equals the card's (the bytes the Phase 6 parity gate validated), and
 *  - the card's feature order equals [Features.NAMES] (what this port computes).
 * Input: the card's window [1, 13, W] built by [Features.make]; outputs "mean", "logvar";
 * the variance is exp(clamped logvar), as in the reference. Batch 1, one intra-op thread.
 * Parity: nav_model_a.json (|kotlin - python| <= 1e-5 + 1e-6 |python|).
 *
 * The ai.onnxruntime API is identical in onnxruntime (JVM, used by tests) and
 * onnxruntime-android (used by the app); :core compiles against it without bundling it.
 */
class ModelIntegrityError(message: String) : RuntimeException(message)

class OnnxSpeedModel(modelBytes: ByteArray, cardJson: String) : SpeedModel, AutoCloseable {
    val name: String
    override val window: Int
    val accelScale: Double
    val gyroScale: Double
    private val inputName: String
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    var lastLatencyMs: Double = Double.NaN
        private set

    init {
        val card = JSONObject(cardJson)
        name = card.optString("name", "model")
        val want = card.getJSONObject("files").getString("onnx_sha256")
        val got = MessageDigest.getInstance("SHA-256").digest(modelBytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
        if (got != want) throw ModelIntegrityError("$name: SHA-256 differs from its model card")
        val input = card.getJSONObject("input")
        val feats = input.getJSONArray("features")
        val names = List(feats.length()) { feats.getString(it) }
        if (names != Features.NAMES) throw ModelIntegrityError("$name: feature order differs from the port's Features.NAMES")
        val shape = input.getJSONArray("shape")
        if (shape.getInt(1) != Features.NAMES.size) throw ModelIntegrityError("$name: expects ${shape.getInt(1)} channels")
        window = shape.getInt(2)
        inputName = input.getString("name")
        val sc = input.getJSONObject("scaling")
        accelScale = sc.getDouble("accel_scale_mps2")
        gyroScale = sc.getDouble("gyro_scale_radps")
        session = env.createSession(modelBytes, OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) })
    }

    /** Raw (mean, logvar) for a ready-made feature tensor (13 x W, channel-major). */
    fun run(features: FloatArray): Pair<Double, Double> {
        require(features.size == 13 * window)
        val t0 = System.nanoTime()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(features), longArrayOf(1, 13, window.toLong())).use { x ->
            session.run(mapOf(inputName to x)).use { r ->
                val mean = (r.get("mean").get().value as FloatArray)[0].toDouble()
                val logvar = (r.get("logvar").get().value as FloatArray)[0].toDouble()
                lastLatencyMs = (System.nanoTime() - t0) / 1e6
                return mean to logvar
            }
        }
    }

    override fun predict(acc: Array<DoubleArray>, grav: Array<DoubleArray>, wz: DoubleArray): Pair<Double, Double> {
        val gyro = Array(wz.size) { doubleArrayOf(0.0, 0.0, wz[it]) }
        val (mean, logvar) = run(Features.make(acc, grav, gyro, accelScale, gyroScale))
        return mean to clampedVariance(logvar)
    }

    override fun close() = session.close()
}
