package com.example.poolwatertester.data

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/** Source of truth for the user's pool profiles. Stored as JSON in
 *  internal filesDir; the active profile id lives in the global
 *  SharedPreferences. On first call we migrate any pre-profile data
 *  (history.json, range_* prefs) into a default profile so an installed
 *  user doesn't lose their existing readings.
 *
 *  Per-profile state is keyed by `id`:
 *    SharedPreferences:  range_<id>_<param>_<kind>
 *    History file:       history_<id>.json
 *  App-global state (reminder, tts, onboarding flag) stays unscoped. */
class ProfilesStore(context: Context) {

    private val appCtx = context.applicationContext
    private val file = File(appCtx.filesDir, "profiles.json")
    private val prefs = appCtx.getSharedPreferences(
        SettingsStore.PREFS_NAME, Context.MODE_PRIVATE)

    init { ensureBootstrap() }

    fun list(): List<Profile> = readJson().let { (profiles, _) -> profiles }

    fun activeId(): String {
        val (profiles, active) = readJson()
        return active ?: profiles.firstOrNull()?.id ?: bootstrapDefault().id
    }

    fun active(): Profile {
        val id = activeId()
        return list().firstOrNull { it.id == id } ?: bootstrapDefault()
    }

    fun create(name: String): Profile {
        val p = Profile(UUID.randomUUID().toString(), name.trim().ifEmpty { "Pool" })
        val (profiles, active) = readJson()
        val updated = profiles + p
        writeJson(updated, active ?: p.id)
        return p
    }

    fun rename(id: String, name: String) {
        val (profiles, active) = readJson()
        val updated = profiles.map { if (it.id == id) it.copy(name = name) else it }
        writeJson(updated, active)
    }

    /** Removes the profile + its per-profile prefs + history file. If the
     *  deleted profile was active, switches to the first remaining one
     *  (or bootstraps a fresh default if the list would be empty). */
    fun delete(id: String) {
        val (profiles, active) = readJson()
        val remaining = profiles.filter { it.id != id }
        clearProfileData(id)
        if (remaining.isEmpty()) {
            val def = Profile(UUID.randomUUID().toString(), "My pool")
            writeJson(listOf(def), def.id)
            return
        }
        val newActive = if (active == id) remaining.first().id else active
        writeJson(remaining, newActive)
    }

    fun setActive(id: String) {
        val (profiles, _) = readJson()
        if (profiles.none { it.id == id }) return
        writeJson(profiles, id)
    }

    // ----------------------------------------------------------- internals

    private fun ensureBootstrap() {
        if (file.exists()) return
        bootstrapDefault()
    }

    /** Creates the default profile and pulls any pre-profile data into it
     *  so old installs keep their history + ranges. */
    private fun bootstrapDefault(): Profile {
        val def = Profile(UUID.randomUUID().toString(), "My pool")
        writeJson(listOf(def), def.id)
        migrateLegacyData(def.id)
        return def
    }

    private fun migrateLegacyData(targetId: String) {
        try {
            val legacyHist = File(appCtx.filesDir, "history.json")
            val newHist = File(appCtx.filesDir, "history_$targetId.json")
            if (legacyHist.exists() && !newHist.exists()) {
                legacyHist.renameTo(newHist)
            }
            val edit = prefs.edit()
            val all = prefs.all
            for ((k, v) in all) {
                if (k.startsWith("range_") && !k.startsWith("range_${targetId}_")) {
                    val newKey = "range_${targetId}_" + k.removePrefix("range_")
                    if (!prefs.contains(newKey) && v is Float) {
                        edit.putFloat(newKey, v)
                        edit.remove(k)
                    }
                }
            }
            edit.apply()
        } catch (e: Exception) {
            Log.w(TAG, "legacy migration failed", e)
        }
    }

    private fun clearProfileData(id: String) {
        try {
            File(appCtx.filesDir, "history_$id.json").delete()
            val edit = prefs.edit()
            val prefix = "range_${id}_"
            for (k in prefs.all.keys) if (k.startsWith(prefix)) edit.remove(k)
            edit.apply()
        } catch (e: Exception) {
            Log.w(TAG, "clearProfileData($id) failed", e)
        }
    }

    private data class State(val profiles: List<Profile>, val active: String?)

    private fun readJson(): State {
        if (!file.exists()) return State(emptyList(), null)
        return try {
            val obj = JSONObject(file.readText())
            val arr = obj.optJSONArray("profiles") ?: JSONArray()
            val profiles = (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                Profile(o.optString("id"), o.optString("name"))
            }
            State(profiles, obj.optString("active").takeIf { it.isNotBlank() })
        } catch (e: Exception) {
            Log.w(TAG, "profiles read failed", e); State(emptyList(), null)
        }
    }

    private fun writeJson(profiles: List<Profile>, active: String?) {
        val obj = JSONObject().apply {
            val arr = JSONArray()
            for (p in profiles) arr.put(JSONObject().apply {
                put("id", p.id); put("name", p.name)
            })
            put("profiles", arr)
            if (active != null) put("active", active)
        }
        val tmp = File(file.parentFile, "${file.name}.tmp")
        tmp.writeText(obj.toString())
        if (!tmp.renameTo(file)) {
            file.delete(); tmp.renameTo(file)
        }
    }

    companion object { const val TAG = "ProfilesStore" }
}
