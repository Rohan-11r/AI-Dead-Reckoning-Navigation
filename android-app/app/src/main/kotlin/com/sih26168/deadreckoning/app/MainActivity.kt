package com.sih26168.deadreckoning.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.sih26168.deadreckoning.app.ui.DiagnosticsScreen
import com.sih26168.deadreckoning.app.ui.NavigationScreen
import com.sih26168.deadreckoning.app.ui.OnboardingScreen
import com.sih26168.deadreckoning.app.ui.RequirementStatus

class MainActivity : ComponentActivity() {
    private val repo get() = (application as NavApplication).repository
    private val prefs get() = getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
        if (granted[Manifest.permission.ACCESS_FINE_LOCATION] == true) {
            startAcquisition()
        } else {
            repo.reportError("precise location was not granted: acquisition not started")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                // onboarding first; the navigation UI only once a name is stored AND every
                // onboarding requirement holds (re-checked at each launch: grants can be revoked)
                var onboarded by rememberSaveable { mutableStateOf(isOnboarded()) }
                if (onboarded) {
                    AppRoot(repo, onStart = ::requestAndStart, onStop = ::stopAcquisition)
                } else {
                    OnboardingScreen(initialName = prefs.getString(KEY_NAME, null).orEmpty()) { name ->
                        prefs.edit().putString(KEY_NAME, name).apply()
                        onboarded = isOnboarded()
                    }
                }
            }
        }
    }

    private fun isOnboarded(): Boolean =
        !prefs.getString(KEY_NAME, null).isNullOrBlank() && RequirementStatus.of(this).allMet

    private fun requestAndStart() {
        val wanted = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            add(Manifest.permission.ACCESS_COARSE_LOCATION)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val missing = wanted.filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) startAcquisition() else permissions.launch(missing.toTypedArray())
    }

    private fun startAcquisition() {
        ContextCompat.startForegroundService(this, Intent(this, AcquisitionService::class.java))
    }

    private fun stopAcquisition() {
        startService(Intent(this, AcquisitionService::class.java).setAction(AcquisitionService.ACTION_STOP))
        repo.outageRequested.value = false
    }

    private companion object {
        const val PREFS = "routeon_onboarding"
        const val KEY_NAME = "user_name"
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppRoot(repo: NavigationRepository, onStart: () -> Unit, onStop: () -> Unit) {
    val nav = rememberNavController()
    val route by nav.currentBackStackEntryAsState()
    val snapshot by repo.snapshot.collectAsStateWithLifecycle()
    val diagnostics by repo.diagnostics.collectAsStateWithLifecycle()
    val geoTrack by repo.geoTrack.collectAsStateWithLifecycle()
    val geoMatchedTrack by repo.geoMatchedTrack.collectAsStateWithLifecycle()
    val running by repo.running.collectAsStateWithLifecycle()
    val outage by repo.outageRequested.collectAsStateWithLifecycle()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                actions = {
                    Row(Modifier.padding(end = 8.dp)) {
                        if (running) Button(onClick = onStop) { Text("Stop") } else Button(onClick = onStart) { Text("Start") }
                    }
                },
            )
        },
        bottomBar = {
            NavigationBar {
                for ((r, label) in listOf("nav" to "Navigation", "diag" to "Diagnostics")) {
                    NavigationBarItem(
                        selected = route?.destination?.route == r,
                        onClick = { nav.navigate(r) { launchSingleTop = true } },
                        icon = {},
                        label = { Text(label) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(nav, startDestination = "nav", modifier = Modifier.padding(padding)) {
            composable("nav") {
                NavigationScreen(snapshot, geoTrack, geoMatchedTrack, diagnostics.hasMap, outage, running,
                    onOutage = { repo.outageRequested.value = it })
            }
            composable("diag") { DiagnosticsScreen(diagnostics) }
        }
    }
}
