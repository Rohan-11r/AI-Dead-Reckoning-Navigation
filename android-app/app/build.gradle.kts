plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.sih26168.deadreckoning.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.sih26168.deadreckoning"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.13.0-phase13"
    }

    buildTypes {
        release {
            isMinifyEnabled = false // no obfuscation until an on-device parity suite exists
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures { compose = true }

    // Assets (task copyExportedModels): models/ = the parity-validated exports + cards (the SAME
    // bytes whose SHA-256 the cards record, checked again at load); config/imu_noise.json =
    // measured process noise; roads/ = the offline road bundle, when one was exported.
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/modelAssets"))
    androidResources { noCompress += listOf("onnx", "bin") }
}

val copyExportedModels by tasks.registering(Sync::class) {
    description = "Bundles the validated models + cards, the measured IMU noise, and any road bundle"
    from(rootProject.file("../models/exported")) {
        include("*.onnx", "*.model_card.json")
        into("models")
    }
    // measured process noise (Phase 4 Allan analysis): the filter must not run on invented Q
    from(rootProject.file("../reports/phase4")) {
        include("imu_noise.json")
        into("config")
    }
    // offline road graph (scripts/export/export_road_bundle.py); optional: absent -> no map matching
    from(rootProject.file("../models/roads")) {
        include("*.roads.bin", "*.roads.json")
        into("roads")
    }
    into(layout.buildDirectory.dir("generated/modelAssets"))
}
tasks.named("preBuild") { dependsOn(copyExportedModels) }

dependencies {
    implementation(project(":core"))
    implementation(project(":sensors"))
    implementation(project(":gnss"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.service)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended) // onboarding logo + permission icons
    implementation(libs.androidx.navigation.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.onnxruntime.android)
    implementation(libs.osmdroid.android) // OpenStreetMap tiles under the track
    debugImplementation(libs.androidx.compose.ui.tooling)

    testImplementation(libs.junit)
}
