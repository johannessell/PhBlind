package com.example.poolwatertester.data

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit

/** SharedPreferences-backed store for per-parameter target ranges + the
 *  periodic reminder configuration. Settings/History fragments observe
 *  changes via `register/unregister` so the UI recolours / reschedules
 *  the moment the user edits a value. */
class SettingsStore(context: Context) {
    private val prefs: SharedPreferences = context.applicationContext
        .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // ------------------------------------------------------------ ranges

    fun rangeFor(param: String): TargetRange? {
        val min = prefs.getFloat(rangeKey(param, "min"), Float.NaN)
        val max = prefs.getFloat(rangeKey(param, "max"), Float.NaN)
        if (min.isNaN() || max.isNaN()) return null
        val ideal = prefs.getFloat(rangeKey(param, "ideal"), (min + max) * 0.5f)
        return TargetRange(min, ideal, max)
    }

    fun setRange(param: String, r: TargetRange) {
        prefs.edit {
            putFloat(rangeKey(param, "min"), r.min)
            putFloat(rangeKey(param, "ideal"), r.ideal)
            putFloat(rangeKey(param, "max"), r.max)
        }
    }

    /** Seed sensible pool defaults for any parameter that has no range
     *  configured yet. The user can edit them in Settings. */
    fun seedDefaultsIfMissing(params: Collection<String>) {
        for (p in params) {
            if (rangeFor(p) != null) continue
            DEFAULT_RANGES[p]?.let { setRange(p, it) }
        }
    }

    // ----------------------------------------------------------- reminder

    var reminderEnabled: Boolean
        get() = prefs.getBoolean(KEY_REMINDER_ENABLED, false)
        set(value) = prefs.edit { putBoolean(KEY_REMINDER_ENABLED, value) }

    var reminderIntervalDays: Int
        get() = prefs.getInt(KEY_REMINDER_INTERVAL_DAYS, 2)
        set(value) = prefs.edit { putInt(KEY_REMINDER_INTERVAL_DAYS, value) }

    var reminderHour: Int
        get() = prefs.getInt(KEY_REMINDER_HOUR, 9)
        set(value) = prefs.edit { putInt(KEY_REMINDER_HOUR, value) }

    var reminderMinute: Int
        get() = prefs.getInt(KEY_REMINDER_MINUTE, 0)
        set(value) = prefs.edit { putInt(KEY_REMINDER_MINUTE, value) }

    // ----------------------------------------------------------- listeners

    fun registerListener(l: SharedPreferences.OnSharedPreferenceChangeListener) {
        prefs.registerOnSharedPreferenceChangeListener(l)
    }

    fun unregisterListener(l: SharedPreferences.OnSharedPreferenceChangeListener) {
        prefs.unregisterOnSharedPreferenceChangeListener(l)
    }

    private fun rangeKey(param: String, kind: String) = "range_${param}_$kind"

    companion object {
        const val PREFS_NAME = "pwt_settings"
        const val KEY_REMINDER_ENABLED = "reminder_enabled"
        const val KEY_REMINDER_INTERVAL_DAYS = "reminder_interval_days"
        const val KEY_REMINDER_HOUR = "reminder_hour"
        const val KEY_REMINDER_MINUTE = "reminder_minute"

        /** Pool-standard defaults; user-editable in Settings. */
        val DEFAULT_RANGES = mapOf(
            "pH"   to TargetRange(7.20f, 7.40f, 7.60f),
            "H2O2" to TargetRange(30.0f, 50.0f, 80.0f),
            "PHMB" to TargetRange(30.0f, 40.0f, 50.0f),
        )

        fun isRangeKey(key: String) = key.startsWith("range_")
        fun paramFromRangeKey(key: String): String? {
            if (!isRangeKey(key)) return null
            val rest = key.removePrefix("range_")
            val lastUnderscore = rest.lastIndexOf('_')
            return if (lastUnderscore < 0) null else rest.substring(0, lastUnderscore)
        }
    }
}
