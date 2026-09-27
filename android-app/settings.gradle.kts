// SIH26168 -- AI-ML dead reckoning, Android app (Phase 10: base & sensors).
// Build machine requirements: JDK 17, Android SDK 35. See README.md.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "sih26168-dead-reckoning"

// :core    -- pure Kotlin/JVM: sample schema, Android->schema mapping, GNSS state machine,
//             rate meters, session logger. No Android dependency: its tests run on any JDK.
// :sensors -- Android SensorManager wrapper (accelerometer, gyroscope, magnetometer, gravity)
// :gnss    -- Android LocationManager / FusedLocationProvider wrapper
// :app     -- foreground acquisition service, Compose UI, diagnostics, logger wiring
include(":core", ":sensors", ":gnss", ":app")
