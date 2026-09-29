package com.sih26168.deadreckoning.app

import android.content.res.AssetManager
import com.sih26168.deadreckoning.core.GnssOnlyEngine
import com.sih26168.deadreckoning.core.NavigationEngine
import com.sih26168.deadreckoning.core.mapmatch.RoadNetwork
import com.sih26168.deadreckoning.core.nav.DeadReckoningNavigation
import com.sih26168.deadreckoning.core.nav.EngineConfig
import com.sih26168.deadreckoning.core.nav.FusionConfig
import com.sih26168.deadreckoning.core.nav.NoiseParams
import com.sih26168.deadreckoning.core.nav.OnnxSpeedModel
import org.json.JSONObject
import java.security.MessageDigest

/**
 * Builds the on-device navigation engine from the APK assets (Gradle task copyExportedModels):
 *  - config/imu_noise.json : the MEASURED process noise (Phase 4). Without it there is no
 *    dead reckoning: the filter must not run on invented Q -> GnssOnlyEngine fallback, said so.
 *  - models/model_a_*.onnx + card : Model A, refused unless its SHA-256 matches its card.
 *    Without it: AI speed OFF, and therefore NHC OFF too (the Phase 7 pairing rule: NHC
 *    alone measured harmful).
 *  - roads/<name>.roads.bin (+ .roads.json manifest) : the offline road graph, SHA-256-checked,
 *    loaded separately (it takes seconds): absent, or more than one -> no map matching, said so.
 * Every downgrade is recorded in [Built.notes] and shown on the diagnostics screen.
 */
object EngineFactory {
    class Built(val engine: NavigationEngine, val dr: DeadReckoningNavigation?, val model: OnnxSpeedModel?, val notes: List<String>)

    fun build(assets: AssetManager): Built {
        val notes = mutableListOf<String>()
        val noise = try {
            loadNoise(assets)
        } catch (x: Exception) {
            notes += "no measured IMU noise (config/imu_noise.json): ${x.message} -> GNSS-only engine"
            return Built(GnssOnlyEngine(), null, null, notes)
        }
        val model = try {
            loadModelA(assets).also { if (it == null) notes += "Model A not bundled -> AI speed and NHC OFF" }
        } catch (x: Exception) {
            notes += "Model A refused (${x.message}) -> AI speed and NHC OFF"
            null
        }
        val dr = DeadReckoningNavigation(noise, EngineConfig(fusion = FusionConfig(useAiSpeed = model != null)), model)
        return Built(dr, dr, model, notes)
    }

    fun loadNoise(assets: AssetManager): NoiseParams {
        val fp = JSONObject(assets.open("config/imu_noise.json").bufferedReader().use { it.readText() }).getJSONObject("filter_parameters")
        return NoiseParams(
            accelWhiteMps2RtHz = fp.getDouble("accel_white_noise_density"),
            gyroWhiteRadpsRtHz = fp.getDouble("gyro_white_noise_density"),
            accelBiasSigmaMps2 = fp.getDouble("accel_bias_sigma"),
            accelBiasTauS = fp.getDouble("accel_bias_tau_s"),
            gyroBiasSigmaRadps = fp.getDouble("gyro_bias_sigma"),
            gyroBiasTauS = fp.getDouble("gyro_bias_tau_s"),
        )
    }

    fun loadModelA(assets: AssetManager): OnnxSpeedModel? {
        val onnx = assets.list("models").orEmpty().filter { it.startsWith("model_a") && it.endsWith(".onnx") }.minOrNull() ?: return null
        val bytes = assets.open("models/$onnx").use { it.readBytes() }
        val card = assets.open("models/${onnx.removeSuffix(".onnx")}.model_card.json").bufferedReader().use { it.readText() }
        return OnnxSpeedModel(bytes, card)
    }

    /** The offline road network, or null (with the reason) when none is bundled or it fails its check. */
    fun loadRoads(assets: AssetManager): Pair<RoadNetwork?, String> {
        val bins = assets.list("roads").orEmpty().filter { it.endsWith(".roads.bin") }.sorted()
        if (bins.size > 1) return null to "${bins.size} road bundles in the APK (${bins.joinToString()}): keep exactly one in models/roads/"
        val bin = bins.firstOrNull()
            ?: return null to "no offline road bundle in the APK (scripts/export/export_road_bundle.py)"
        val bytes = assets.open("roads/$bin").use { it.readBytes() }
        val manifest = runCatching {
            JSONObject(assets.open("roads/${bin.removeSuffix(".roads.bin")}.roads.json").bufferedReader().use { it.readText() })
        }.getOrNull() ?: return null to "$bin has no manifest: refused"
        val got = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
        if (got != manifest.getString("sha256")) return null to "$bin: SHA-256 differs from its manifest: refused"
        val net = bytes.inputStream().use { RoadNetwork.read(it) }
        return net to "$bin: ${net.nSegments} directed segments"
    }
}
