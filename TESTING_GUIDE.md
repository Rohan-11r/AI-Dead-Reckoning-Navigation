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
- [ ] Git; **Python 3.14** for step 5 (road bundle; optionally step 4)
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

## 5. Road bundle for your test area (map matching)

Map matching needs an offline road graph of the area you will drive, built **before** the APK
(the build copies `models/roads/` into the app). The bundle is not in Git (it is derived OSM
data, ODbL); you generate it once per area. The only bundle in the repo's history covers
Coventry, UK — useless in Nagpur. Needs internet on the build machine only; the app itself
never downloads maps.

**Python for this step:** only numpy and the `navcore` package — no torch:

```bash
python3.14 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy
pip install -e . --no-deps
```

(If you already did step 4, that environment works as is.)

**Generate the bundle** from the repo root, either by city name or by bounding box:

```bash
# (a) city name: uses the bounding box of the first OpenStreetMap (Nominatim) match
python scripts/export/export_road_bundle.py --city "Nagpur, India"

# (b) bounding box: SOUTH WEST NORTH EAST in decimal degrees, plus a name for the files
python scripts/export/export_road_bundle.py --name nagpur --bbox 21.05 79.00 21.23 79.18
```

Use (b) when your routes leave the city's administrative box. For "Nagpur, India", (a) resolved
on 2026-09-28 to *Nagpur City* with box S 21.053, W 78.995, N 21.231, E 79.179 (about
20 × 19 km). That misses the outskirts, e.g. Hingna, Butibori and most of the Outer Ring
Road. To read off a box, open openstreetmap.org, click **Export** → *Manually select a
different area*, and drag it over your routes; the four numbers are shown there. Keep each
side under 0.6° (≈ 65 km). Larger boxes are refused, since the public server times out and the
bundle gets too big for a phone. `--allow-large` overrides this.

What it does: downloads the drivable roads once through the Overpass API (the same query as
the Coventry bundle), then writes

| File | Tracked in Git? | What |
| --- | --- | --- |
| `data/raw/osm/<name>_drivable.osm` | no | the raw OSM download (cached; re-runs reuse it) |
| `reports/osm/<name>_osm_manifest.json` | yes, if you commit it | provenance: query, box, time, SHA-256 |
| `models/roads/<name>.roads.bin` | no | **the bundle the app loads** |
| `models/roads/<name>.roads.json` | yes, if you commit it | its manifest; the app refuses the `.bin` unless the SHA-256 matches |

The (a) run above took about 2.5 minutes on 2026-09-28: a 15.6 MB download, then a 7.8 MB
bundle with 279,967 directed segments. If Overpass answers with an HTTP 429 or 504, the public
server is busy; wait a few minutes and re-run. To change the box for an existing name,
delete that area's `.osm` file and its `reports/osm/` manifest first; the script refuses to
mix two boxes under one name.

- [ ] Script printed `wrote models/roads/<name>.roads.bin (... segments)`
- [ ] **Exactly one** `*.roads.bin` in `models/roads/`. The app loads a single bundle; with two
      or more it refuses to choose and runs without map matching. The script warns you if
      another one (e.g. `coventry.roads.bin` from the repo owner) is lying there; move it out.
- [ ] Then build the app (step 6). Rebuild whenever you regenerate the bundle.

Limits, so the results are read correctly: map matching is **refinement only**. It never
produces a position, and with it removed the dead-reckoned track must still stand
(AGENTS.md §1). On the Coventry validation drives it gave only a **marginal** gain, and only
while drift stayed under about 30 m (PROJECT_STATUS.md Phase 8). It has never been run on
Indian roads, where OSM completeness and one-way tagging may differ. Loading and memory use of
a bundle on a phone have not been measured.

## 6. Build and install the app

```bash
cd android-app
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

- [ ] APK at `app/build/outputs/apk/debug/app-debug.apk` (CI also uploads it as the `app-debug-apk` artifact)
- [ ] Installs and launches; grant location + notification permissions

**Map matching:** the APK contains the road bundle only if one was in `models/roads/` when you
built (step 5). Without one — a fresh checkout, and every CI build — the app runs with **no map
matching** and says so under "Engine resources".

## 7. On the handset (Phase 13 checks, start of Phase 14)

Record what you observe, including failures — every number goes in PROJECT_STATUS.md only
with the session log it came from.

- [ ] **Diagnostics** screen: accelerometer / gyroscope / magnetometer / gravity rates look
      plausible for the phone; rejection / drop counters stay near zero
- [ ] "Engine resources" lists the noise file and Model A as loaded (else dead reckoning is
      downgraded, and the screen says why), and the road bundle as `<name>.roads.bin: N directed
      segments` (else no map matching, with the reason)
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

## 8. Report back

- [ ] Build result (pass / the first error), `:core:test` result (the HTML report)
- [ ] Any source change you had to make, as a commit with an explanation
- [ ] Session folders from the drives, plus your notes from step 7
