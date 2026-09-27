package com.sih26168.deadreckoning.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import org.json.JSONObject
import java.nio.FloatBuffer
import java.security.MessageDigest

/**
 * Measures on-device inference time of the bundled speed model (Model A, Phase 6 export).
 *
 * What it measures, stated exactly: batch-1, single-thread ONNX Runtime CPU inference on an
 * ALL-ZERO window of the model card's shape [1, 13, W]. That is COMPUTE LATENCY ONLY: the
 * on-device feature pipeline (navcore.fusion.features) is not ported yet, so the model is
 * not driving navigation, and this number says nothing about its accuracy. (For these
 * conv/recurrent graphs, latency does not depend on the input values.)
 *
 * Integrity: the model is refused if its SHA-256 differs from the one its card records --
 * the same check the Python runtime makes (navcore.fusion.ai_models.OnnxSequenceModel).
 */
class OnnxLatencyProbe(private val assets: AssetManager) {
    fun measure(runs: Int = 200, warmup: Int = 20): AiLatency {
        val names = assets.list("models").orEmpty()
        val onnx = names.filter { it.startsWith("model_a") && it.endsWith(".onnx") }.minOrNull()
            ?: return AiLatency("none", null, null, 0, "no Model A in assets/models (build task copyExportedModels)")
        val bytes = assets.open("models/$onnx").use { it.readBytes() }
        val card = JSONObject(assets.open("models/${onnx.removeSuffix(".onnx")}.model_card.json").bufferedReader().use { it.readText() })
        val want = card.getJSONObject("files").getString("onnx_sha256")
        val got = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
        if (got != want) return AiLatency(onnx, null, null, 0, "REFUSED: SHA-256 differs from its model card")
        val input = card.getJSONObject("input")
        val shape = input.getJSONArray("shape")
        val c = shape.getInt(1)
        val w = shape.getInt(2)
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
        env.createSession(bytes, opts).use { session ->
            OnnxTensor.createTensor(env, FloatBuffer.allocate(c * w), longArrayOf(1, c.toLong(), w.toLong())).use { x ->
                val feed = mapOf(input.getString("name") to x)
                repeat(warmup) { session.run(feed).close() }
                val ms = DoubleArray(runs) {
                    val t0 = System.nanoTime()
                    session.run(feed).close()
                    (System.nanoTime() - t0) / 1e6
                }.sorted()
                return AiLatency(onnx, ms[runs / 2], ms[runs * 9 / 10], runs,
                    "batch 1, 1 thread, all-zero input: compute latency only (features not ported yet)")
            }
        }
    }
}
