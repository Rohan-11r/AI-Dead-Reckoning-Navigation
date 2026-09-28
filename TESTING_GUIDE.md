# TESTING_GUIDE.md — build machine & handset checklist

For the teammate building and field-testing the Android port (PROJECT_STATUS.md Phases 12–14).
The Kotlin in `android-app/` was written on a machine with **no JDK or Android SDK**: it has
**never been compiled and none of its tests has run**. Your first build is the first build.
Build errors are expected work; please report them rather than work around them silently.

CI (`.github/workflows/android.yml`) runs the same commands on every push; the steps below
are for running them locally and on a phone.

Commands are shown for Linux/macOS/Git Bash. On Windows `cmd`/PowerShell use `gradlew.bat`
in place of `./gradlew`.

---

## 0. Prerequisites

- [ ] **JDK 17** — `java -version` prints `17.x`
- [ ] **Android SDK**: Platform 35 + Build-Tools (Android Studio installs both); set
      `ANDROID_HOME`, or create `android-app/local.properties` with `sdk.dir=/path/to/Android/sdk`
- [ ] **Gradle 8.11.1** — only for the one-time wrapper step below (skip if `android-app/gradlew` exists)
- [ ] Git; optionally **Python 3.14** for step 4
- [ ] A handset with Android 8.0+ (API 26), USB debugging on, `adb devices` lists it

## 1. Get the code

```bash
git clone <repo-url> sih26168 && cd sih26168
```

Build **inside the full monorepo checkout**: the Kotlin tests read `tests/regression/golden/`
and `models/exported/`, and the APK bundles `models/exported/` and `reports/phase4/imu_noise.json`.

## 2. Gradle wrapper (once)

The wrapper JAR is not committed yet. If `android-app/gradlew` is missing:

```bash
cd android-app
gradle wrapper --gradle-version 8.11.1
git add gradlew gradlew.bat gradle/wrapper/gradle-wrapper.jar
git commit -m "Add Gradle wrapper (8.11.1)"
```

- [ ] `./gradlew --version` prints Gradle 8.11.1
- [ ] Wrapper committed and pushed (so CI and everyone else use it)

If dependency resolution fails, the versions in `gradle/libs.versions.toml` may need bumping;
note what you changed in the commit message.

## 3. Numerical parity — `:core` tests (pure JVM, no Android SDK needed)

```bash
cd android-app
./gradlew :core:test
```

- [ ] Passes: 37 tests in 5 classes (`CoreTest`, `MappingTest`, `DisplayTest`, `ParityTest`, `nav.NavParityTest`)
- [ ] Report: `core/build/reports/tests/test/index.html`

Only the parity tests, if you need to iterate:

```bash
./gradlew :core:test --tests "com.sih26168.deadreckoning.core.ParityTest"
./gradlew :core:test --tests "com.sih26168.deadreckoning.core.nav.NavParityTest"
```

**If a parity test fails:** do **not** loosen a tolerance or edit a golden file to make it pass
(AGENTS.md §4: a parity failure is a release blocker). Send the test name, the failing vector,
and the expected vs actual values from the report; the fix belongs in the Kotlin port (or, if
the reference is wrong, in the Python reference followed by regenerating the vectors).

## 4. (Optional) Python reference side

Checks that the committed golden vectors are still what the Python reference produces.

```bash
python3.14 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest tests/regression/test_golden_vectors.py -v
```

- [ ] Passes. The vectors were generated on Windows; a mismatch on another OS has not been
      ruled out and is worth reporting, not ignoring.

## 5. Build and install the app

```bash
cd android-app
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

- [ ] APK at `app/build/outputs/apk/debug/app-debug.apk` (CI also uploads it as the `app-debug-apk` artifact)
- [ ] Installs and launches; grant location + notification permissions

**Map matching:** the road bundle (`models/roads/coventry.roads.bin`, 13.2 MB) is gitignored,
so a fresh checkout — and every CI build — has **no map matching**; the app says so under
"Engine resources". Get the file from the repo owner out of band and place it in
`models/roads/` before building; its SHA-256 must equal the one in
`models/roads/coventry.roads.json`. It covers Coventry (UK) only: anywhere else there is no
road network to match to, whatever the build.

## 6. On the handset (Phase 13 checks, start of Phase 14)

Record what you observe, including failures — every number goes in PROJECT_STATUS.md only
with the session log it came from.

- [ ] **Diagnostics** screen: accelerometer / gyroscope / magnetometer / gravity rates look
      plausible for the phone; rejection / drop counters stay near zero
- [ ] "Engine resources" lists the noise file and Model A as loaded (else dead reckoning is
      downgraded, and the screen says why)
- [ ] Outdoors with a GNSS fix: state reaches `GOOD`; the track follows the road
- [ ] Toggle **Simulate GNSS outage** while driving: state goes to `DEAD_RECKONING`, the red
      marker **keeps moving**; toggle off: GNSS is re-fused (a visible jump is expected)
- [ ] A real outage (tunnel / underpass / multi-storey car park), if available
- [ ] Session logs pulled for analysis:
      ```bash
      adb pull /sdcard/Android/data/com.sih26168.deadreckoning/files/sessions/ ./sessions/
      ```
      Each drive: `manifest.json`, `sensors.csv`, `gnss.csv`, `events.csv`
- [ ] Note the phone model and Android version, and how the phone was mounted

Not yet measurable from the app (known gaps, PROJECT_STATUS.md §13C): battery / CPU /
thermals, doze behaviour, map tiles, an on-device golden-recording replay.

## 7. Report back

- [ ] Build result (pass / the first error), `:core:test` result (the HTML report)
- [ ] Any source change you had to make, as a commit with an explanation
- [ ] Session folders from the drives, plus your notes from step 6
