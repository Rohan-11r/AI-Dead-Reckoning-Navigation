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
        versionName = "0.10.0-phase10"
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
    buildFeatures { compose = true }

    // The exported, parity-validated models (Phase 6) and their cards ship as assets under
    // models/: the SAME bytes whose SHA-256 the cards record (checked again at load).
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/modelAssets"))
    androidResources { noCompress += "onnx" }
}

kotlin { jvmToolchain(17) }

val copyExportedModels by tasks.registering(Copy::class) {
    description = "Copies models/exported/*.onnx + model cards into the APK assets"
    from(rootProject.file("../models/exported")) {
        include("*.onnx", "*.model_card.json")
    }
    into(layout.buildDirectory.dir("generated/modelAssets/models"))
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
    implementation(libs.androidx.navigation.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.onnxruntime.android)
    debugImplementation(libs.androidx.compose.ui.tooling)

    testImplementation(libs.junit)
}
