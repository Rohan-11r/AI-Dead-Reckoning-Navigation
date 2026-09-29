// Root build: plugin versions come from gradle/libs.versions.toml; nothing is applied here.
// Module JVM targets (Java 17 / JVM target 17) are configured directly in subprojects (:app, :core, :gnss, :sensors).
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
