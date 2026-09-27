package com.sih26168.deadreckoning.core

import java.util.Locale

/**
 * GNSS quality state machine: a line-by-line port of `navcore.state.gnss_state` (Python
 * reference). Parity is enforced by golden vectors generated from the reference
 * (tests/regression/golden/gnss_state_machine.json, replayed in GnssStateMachineTest).
 *
 *   GOOD       -> DEGRADED    accuracy > degradedAccuracyM, OR a fix rejected by the gate,
 *                             OR fix age > degradedAgeS
 *   GOOD/DEG.  -> LOST        fix age > lostAgeS
 *   DEGRADED   -> GOOD        goodStreak consecutive accepted, accurate fixes
 *   LOST       -> RECOVERING  first fix after LOST
 *   RECOVERING -> GOOD        recoverStreak consecutive accepted, accurate fixes
 *   RECOVERING -> LOST        fix age > lostAgeS again
 *   RECOVERING -> DEGRADED    an inaccurate or rejected fix
 *
 * The UI shows LOST as "DEAD_RECKONING" ([displayName]); the state itself keeps the
 * reference's name so the two implementations compare one to one.
 */
enum class GnssState(val displayName: String) {
    GOOD("GOOD"),
    DEGRADED("DEGRADED"),
    LOST("DEAD_RECKONING"),
    RECOVERING("RECOVERING"),
}

data class GnssStateConfig(
    val degradedAccuracyM: Double = 10.0,
    val degradedAgeS: Double = 12.0,
    val lostAgeS: Double = 25.0,
    val goodStreak: Int = 2,
    val recoverStreak: Int = 3,
)

data class FilterPolicy(
    val useGnss: Boolean,
    val gnssNoiseInflation: Double,
    val useAiSpeed: Boolean,
    val useNhc: Boolean,
)

val POLICY: Map<GnssState, FilterPolicy> = mapOf(
    GnssState.GOOD to FilterPolicy(useGnss = true, gnssNoiseInflation = 1.0, useAiSpeed = false, useNhc = true),
    GnssState.DEGRADED to FilterPolicy(useGnss = true, gnssNoiseInflation = 2.0, useAiSpeed = true, useNhc = true),
    GnssState.LOST to FilterPolicy(useGnss = false, gnssNoiseInflation = 1.0, useAiSpeed = true, useNhc = true),
    GnssState.RECOVERING to FilterPolicy(useGnss = true, gnssNoiseInflation = 3.0, useAiSpeed = true, useNhc = true),
)

data class Transition(val tS: Double, val from: GnssState, val to: GnssState, val why: String)

class GnssStateMachine(val cfg: GnssStateConfig = GnssStateConfig()) {
    var state: GnssState = GnssState.LOST  // nothing is trusted before the first fix
        private set
    var lastFixT: Double? = null
        private set
    var streak: Int = 0
        private set
    val transitions = mutableListOf<Transition>()

    private fun go(t: Double, new: GnssState, why: String) {
        if (new != state) {
            transitions += Transition(t, state, new, why)
            state = new
            streak = 0
        }
    }

    /** Advance with no fix: age-based downgrades only. */
    fun onTime(t: Double): GnssState {
        val last = lastFixT
        val age = if (last == null) Double.POSITIVE_INFINITY else t - last
        if (state != GnssState.LOST && age > cfg.lostAgeS) {
            go(t, GnssState.LOST, fmt1("no fix for %.1f s", age))
        } else if (state == GnssState.GOOD && age > cfg.degradedAgeS) {
            go(t, GnssState.DEGRADED, fmt1("fix age %.1f s", age))
        }
        return state
    }

    /** Update with a received fix and the filter's gate outcome for it. */
    fun onFix(t: Double, accuracyM: Double, accepted: Boolean): GnssState {
        lastFixT = t
        val good = accepted && accuracyM <= cfg.degradedAccuracyM
        val why = when {
            good -> "accurate fix accepted"
            !accepted -> "fix rejected by gate"
            else -> fmt1("accuracy %.1f m", accuracyM)
        }
        when (state) {
            GnssState.LOST -> {
                go(t, GnssState.RECOVERING, "first fix after loss")
                streak = if (good) 1 else 0
            }
            GnssState.GOOD -> if (!good) go(t, GnssState.DEGRADED, why)
            GnssState.DEGRADED -> {
                streak = if (good) streak + 1 else 0
                if (streak >= cfg.goodStreak) go(t, GnssState.GOOD, "${cfg.goodStreak} good fixes")
            }
            GnssState.RECOVERING -> if (!good) {
                go(t, GnssState.DEGRADED, why)
            } else {
                streak += 1
                if (streak >= cfg.recoverStreak) go(t, GnssState.GOOD, "${cfg.recoverStreak} good fixes after loss")
            }
        }
        return state
    }

    val policy: FilterPolicy get() = POLICY.getValue(state)
}

/** Locale-independent formatting: a device set to, e.g., German must log "12.0", not "12,0". */
internal fun fmt1(pattern: String, x: Double): String = String.format(Locale.ROOT, pattern, x)
