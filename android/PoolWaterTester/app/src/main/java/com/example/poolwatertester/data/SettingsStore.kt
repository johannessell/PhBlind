package com.example.poolwatertester.data

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit

/** SharedPreferences-backed store. Per-parameter target ranges are scoped
 *  per profile (`range_<profileId>_<param>_<kind>`); reminder + TTS +
 *  onboarding state stay app-global. Callers either pass in a profile id
 *  explicitly or use [forActive] to bind to the currently-active one. */
class SettingsStore(context: Context, val profileId: String) {

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

    /** Whether the parameter should appear in History charts/stats. The
     *  reading is still measured and saved when present in the reference —
     *  this only controls display. Default: shown. */
    fun isParamShown(param: String): Boolean =
        prefs.getBoolean(showKey(param), true)

    fun setParamShown(param: String, shown: Boolean) {
        prefs.edit { putBoolean(showKey(param), shown) }
    }

    /** Seed sensible pool defaults for any parameter that has no range
     *  configured yet (for the bound profile). User-editable in Settings. */
    fun seedDefaultsIfMissing(params: Collection<String>) {
        for (p in params) {
            if (rangeFor(p) != null) continue
            DEFAULT_RANGES[p]?.let { setRange(p, it) }
        }
    }

    // -------------------------------------------------------- app-global

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

    var ttsEnabled: Boolean
        get() = prefs.getBoolean(KEY_TTS_ENABLED, false)
        set(value) = prefs.edit { putBoolean(KEY_TTS_ENABLED, value) }

    var onboardingDone: Boolean
        get() = prefs.getBoolean(KEY_ONBOARDING_DONE, false)
        set(value) = prefs.edit { putBoolean(KEY_ONBOARDING_DONE, value) }

    var activeProfileId: String?
        get() = prefs.getString(KEY_ACTIVE_PROFILE_ID, null)
        set(value) = prefs.edit { putString(KEY_ACTIVE_PROFILE_ID, value) }

    // ----------------------------------------------------------- listeners

    fun registerListener(l: SharedPreferences.OnSharedPreferenceChangeListener) {
        prefs.registerOnSharedPreferenceChangeListener(l)
    }

    fun unregisterListener(l: SharedPreferences.OnSharedPreferenceChangeListener) {
        prefs.unregisterOnSharedPreferenceChangeListener(l)
    }

    private fun rangeKey(param: String, kind: String) =
        "range_${profileId}_${param}_$kind"

    private fun showKey(param: String) = "show_${profileId}_$param"

    companion object {
        const val PREFS_NAME = "pwt_settings"
        const val KEY_REMINDER_ENABLED = "reminder_enabled"
        const val KEY_REMINDER_INTERVAL_DAYS = "reminder_interval_days"
        const val KEY_REMINDER_HOUR = "reminder_hour"
        const val KEY_REMINDER_MINUTE = "reminder_minute"
        const val KEY_TTS_ENABLED = "tts_enabled"
        const val KEY_ONBOARDING_DONE = "onboarding_done"
        const val KEY_ACTIVE_PROFILE_ID = "active_profile_id"

        /** Pool-standard defaults; user-editable in Settings. */
        val DEFAULT_RANGES = mapOf(
            "pH"   to TargetRange(7.20f, 7.40f, 7.60f),
            "H2O2" to TargetRange(30.0f, 50.0f, 80.0f),
            "PHMB" to TargetRange(30.0f, 40.0f, 50.0f),
        )

        /** Convenience: bind to the profile that ProfilesStore says is active. */
        fun forActive(context: Context): SettingsStore {
            val id = ProfilesStore(context).activeId()
            return SettingsStore(context, id)
        }
    }
}
