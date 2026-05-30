package com.example.poolwatertester

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.fragment.app.Fragment
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.poolwatertester.databinding.ActivityMainBinding
import com.example.poolwatertester.ui.HistoryFragment
import com.example.poolwatertester.ui.MeasureFragment
import com.example.poolwatertester.ui.SettingsFragment

/** Host activity. Owns the BottomNavigationView, starts Python once, and
 *  switches between Measure / History / Settings fragments. All camera +
 *  measurement behaviour lives in [MeasureFragment]; this activity is
 *  deliberately thin so future tabs can be added without touching the
 *  measure flow. */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.main) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))

        binding.bottomNav.setOnItemSelectedListener { item ->
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
            binding.bottomNav.selectedItemId = R.id.nav_measure
        }
    }
}
