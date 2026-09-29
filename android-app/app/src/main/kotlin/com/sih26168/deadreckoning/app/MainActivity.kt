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
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
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
import com.sih26168.deadreckoning.app.ui.NameScreen
import com.sih26168.deadreckoning.app.ui.NavigationScreen
import com.sih26168.deadreckoning.app.ui.PermissionsScreen
import com.sih26168.deadreckoning.app.ui.RequirementStatus
import com.sih26168.deadreckoning.app.ui.SplashScreen
import com.sih26168.deadreckoning.app.ui.WelcomeTransitionScreen
import com.sih26168.deadreckoning.app.ui.theme.RouteonTheme

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
            RouteonTheme { RootGraph() }
        }
    }

    /**
     * Splash -> WelcomeTransition -> Name -> Permissions -> Map. The splash runs on EVERY launch;
     * after it, a returning user whose name is stored and whose grants all still hold goes straight
     * to the map (re-checked at each launch: grants can be revoked), one with a stored name but a
     * revoked grant goes to Permissions, anyone else starts at the welcome screen.
     */
    @Composable
    private fun RootGraph() {
        val root = rememberNavController()
        // the map replaces the whole onboarding back stack: Back from the map leaves the app
        fun toMap() = root.navigate(Route.MAP) { popUpTo(root.graph.id) { inclusive = true } }
        NavHost(root, startDestination = Route.SPLASH) {
            composable(Route.SPLASH) {
                SplashScreen(onFinished = {
                    val next = when {
                        isOnboarded() -> Route.MAP
                        !storedName().isNullOrBlank() -> Route.PERMISSIONS
                        else -> Route.WELCOME
                    }
                    root.navigate(next) { popUpTo(Route.SPLASH) { inclusive = true } }
                })
            }
            composable(Route.WELCOME) { WelcomeTransitionScreen(onStart = { root.navigate(Route.NAME) { launchSingleTop = true } }) }
            composable(Route.NAME) {
                NameScreen(initialName = storedName().orEmpty(), onNext = { name ->
                    prefs.edit().putString(KEY_NAME, name).apply()
                    root.navigate(Route.PERMISSIONS) { launchSingleTop = true }
                })
            }
            composable(Route.PERMISSIONS) {
                PermissionsScreen(onComplete = {
                    if (isOnboarded()) toMap() else root.navigate(Route.NAME) { launchSingleTop = true } // name missing: ask for it
                })
            }
            composable(Route.MAP) { AppRoot(repo, onStart = ::requestAndStart, onStop = ::stopAcquisition) }
        }
    }

    private fun storedName(): String? = prefs.getString(KEY_NAME, null)

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

/** Top-level routes, in onboarding order. */
private object Route {
    const val SPLASH = "splash"
    const val WELCOME = "welcome"
    const val NAME = "name"
    const val PERMISSIONS = "permissions"
    const val MAP = "map"
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
    // the map runs full-screen under the top bar; "nav" is also the start route (null before the first frame)
    val onMap = route?.destination?.route.let { it == null || it == "nav" }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                colors = if (onMap) {
                    TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                } else TopAppBarDefaults.topAppBarColors(),
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
        // padding per route: the map draws under the bars, its floating panels take the padding
        NavHost(nav, startDestination = "nav") {
            composable("nav") {
                NavigationScreen(snapshot, geoTrack, geoMatchedTrack, diagnostics.hasMap, diagnostics.alignment, outage, running,
                    onOutage = { repo.outageRequested.value = it }, contentPadding = padding)
            }
            composable("diag") { DiagnosticsScreen(diagnostics, Modifier.padding(padding)) }
        }
    }
}
