package com.example.poolwatertester.data

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** One persisted measurement: the time, the per-parameter measured values,
 *  and the relative path to the per-measurement log folder (input.jpg /
 *  overlay.jpg / result.txt). `logDir` lets the History detail sheet show
 *  the overlay image without indexing it separately. */
data class HistoryEntry(
    val ts: Long,
    val results: Map<String, Float>,
    val logDir: String,
)

/** Append-only JSON file per profile (`filesDir/history_<profileId>.json`).
 *  Atomic via tmp + rename so a crash mid-write can never produce a
 *  half-file. Sized for hundreds of entries — swap for Room/SQLite if it
 *  ever needs thousands. */
class HistoryStore(context: Context, val profileId: String) {
    private val file = File(
        context.applicationContext.filesDir, "history_$profileId.json")

    fun append(entry: HistoryEntry) {
        try {
            val arr = readArray()
            arr.put(entry.toJson())
            writeArrayAtomic(arr)
        } catch (e: Exception) {
            Log.w(TAG, "history append failed", e)
        }
    }

    fun loadAll(): List<HistoryEntry> {
        return try {
            val arr = readArray()
            (0 until arr.length()).mapNotNull { i ->
                arr.optJSONObject(i)?.toHistoryEntry()
            }.sortedBy { it.ts }
        } catch (e: Exception) {
            Log.w(TAG, "history load failed", e); emptyList()
        }
    }

    /** Removes any entry with the given ts. Returns true iff something
     *  changed (so the caller can skip a redraw). */
    fun deleteByTs(ts: Long): Boolean {
        return try {
            val arr = readArray()
            var removed = false
            val out = JSONArray()
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                if (o.optLong("ts", 0L) == ts) { removed = true; continue }
                out.put(o)
            }
            if (removed) writeArrayAtomic(out)
            removed
        } catch (e: Exception) {
            Log.w(TAG, "history deleteByTs failed", e); false
        }
    }

    /** Empties the history for this profile. Returns how many entries
     *  were removed. */
    fun clearAll(): Int {
        return try {
            val n = readArray().length()
            writeArrayAtomic(JSONArray())
            n
        } catch (e: Exception) {
            Log.w(TAG, "history clearAll failed", e); 0
        }
    }

    /** Bulk import (used by CSV restore). Idempotent by ts — entries
     *  whose ts already exists are skipped. Returns how many were added. */
    fun appendAllSkippingDuplicates(entries: List<HistoryEntry>): Int {
        return try {
            val arr = readArray()
            val seen = HashSet<Long>()
            for (i in 0 until arr.length()) {
                val ts = arr.optJSONObject(i)?.optLong("ts", 0L) ?: continue
                seen.add(ts)
            }
            var added = 0
            for (e in entries) {
                if (seen.add(e.ts)) { arr.put(e.toJson()); added++ }
            }
            if (added > 0) writeArrayAtomic(arr)
            added
        } catch (e: Exception) {
            Log.w(TAG, "history bulk import failed", e); 0
        }
    }

    // ----------------------------------------------------------- helpers

    private fun HistoryEntry.toJson(): JSONObject = JSONObject().apply {
        put("ts", ts)
        put("logDir", logDir)
        val r = JSONObject()
        for ((k, v) in results) r.put(k, v.toDouble())
        put("results", r)
    }

    private fun JSONObject.toHistoryEntry(): HistoryEntry? {
        val ts = optLong("ts", 0L); if (ts == 0L) return null
        val logDir = optString("logDir", "")
        val r = optJSONObject("results") ?: JSONObject()
        val results = mutableMapOf<String, Float>()
        val it = r.keys()
        while (it.hasNext()) {
            val k = it.next()
            val v = r.optDouble(k, Double.NaN).toFloat()
            if (!v.isNaN()) results[k] = v
        }
        return HistoryEntry(ts, results.toMap(), logDir)
    }

    private fun readArray(): JSONArray {
        if (!file.exists()) return JSONArray()
        val text = file.readText()
        if (text.isBlank()) return JSONArray()
        return JSONArray(text)
    }

    private fun writeArrayAtomic(arr: JSONArray) {
        val tmp = File(file.parentFile, "${file.name}.tmp")
        tmp.writeText(arr.toString())
        if (!tmp.renameTo(file)) {
            file.delete()
            if (!tmp.renameTo(file)) tmp.copyTo(file, overwrite = true)
        }
    }

    companion object {
        const val TAG = "HistoryStore"

        /** Convenience: bind to the profile that ProfilesStore says is active. */
        fun forActive(context: Context): HistoryStore {
            val id = ProfilesStore(context).activeId()
            return HistoryStore(context, id)
        }
    }
}
