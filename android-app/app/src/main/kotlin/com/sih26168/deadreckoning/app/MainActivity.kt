package com.sih26168.deadreckoning.app

import android.Manifest
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.sih26168.deadreckoning.app.ui.DiagnosticsScreen
import com.sih26168.deadreckoning.app.ui.NavigationScreen

class MainActivity : ComponentActivity() {
    private val repo get() = (application as NavApplication).repository

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
                AppRoot(repo, onStart = ::requestAndStart, onStop = ::stopAcquisition)
            }
        }
    }

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
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppRoot(repo: NavigationRepository, onStart: () -> Unit, onStop: () -> Unit) {
    val nav = rememberNavController()
    val route by nav.currentBackStackEntryAsState()
    val snapshot by repo.snapshot.collectAsStateWithLifecycle()
    val diagnostics by repo.diagnostics.collectAsStateWithLifecycle()
    val track by repo.track.collectAsStateWithLifecycle()
    val matchedTrack by repo.matchedTrack.collectAsStateWithLifecycle()
    val running by repo.running.collectAsStateWithLifecycle()
    val outage by repo.outageRequested.collectAsStateWithLifecycle()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("SIH26168 DR") },
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
                NavigationScreen(snapshot, track, matchedTrack, diagnostics.hasMap, outage, running,
                    onOutage = { repo.outageRequested.value = it })
            }
            composable("diag") { DiagnosticsScreen(diagnostics) }
        }
    }
}
