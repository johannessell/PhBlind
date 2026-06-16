package com.example.poolwatertester

import android.content.Intent
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.fragment.app.Fragment
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.poolwatertester.data.ProfilesStore
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.databinding.ActivityMainBinding
import com.example.poolwatertester.ui.HistoryFragment
import com.google.android.material.chip.Chip
import com.google.android.material.navigation.NavigationBarView
import com.example.poolwatertester.ui.MeasureFragment
import com.example.poolwatertester.ui.OnboardingActivity
import com.example.poolwatertester.ui.ProfileSwitchBottomSheet
import com.example.poolwatertester.ui.SettingsFragment

/** Host activity. Owns the BottomNavigationView, the profile chip, the
 *  Python runtime startup, and the first-run onboarding gate. All camera
 *  + measurement behaviour lives in [MeasureFragment]. */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // First-run gate: send the user through the onboarding pager
        // before we ever inflate the main UI. We bind through forActive so
        // the bootstrap migration runs and we get a default profile.
        val settings = SettingsStore.forActive(this)
        if (!settings.onboardingDone) {
            startActivity(Intent(this, OnboardingActivity::class.java))
            finish()
            return
        }

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.main) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))

        refreshProfileChip()
        profileChip()?.setOnClickListener {
            ProfileSwitchBottomSheet.newInstance(onSwitched = { recreate() })
                .show(supportFragmentManager, "profile-switch")
        }

        val nav = findViewById<NavigationBarView>(R.id.bottomNav)
        nav.setOnItemSelectedListener { item ->
            val frag: Fragment = when (item.itemId) {
                R.id.nav_measure  -> MeasureFragment()
                R.id.nav_history  -> HistoryFragment()
                R.id.nav_settings -> SettingsFragment()
                else -> return@setOnItemSelectedListener false
            }
            supportFragmentManager.beginTransaction()
                .setReorderingAllowed(true)
                .replace(R.id.navHost, frag)
                .commit()
            true
        }

        if (savedInstanceState == null) {
            nav.selectedItemId = R.id.nav_measure
        }
    }

    override fun onResume() {
        super.onResume()
        if (this::binding.isInitialized) refreshProfileChip()
    }

    /** Profile chip lives in the portrait `topBar` OR the landscape rail
     *  header — find it from the activity root so both variants resolve. */
    private fun profileChip(): Chip? = findViewById(R.id.profileChip)

    private fun refreshProfileChip() {
        profileChip()?.text = ProfilesStore(this).active().name
    }
}
