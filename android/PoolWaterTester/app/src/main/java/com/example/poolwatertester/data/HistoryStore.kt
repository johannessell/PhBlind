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

/** Append-only JSON file in internal `filesDir/history.json`. Atomic via
 *  tmp + rename so a crash mid-write can never produce a half-file. Sized
 *  for hundreds of entries — if this ever needs thousands, swap for a
 *  Room/SQLite store without changing callers. */
class HistoryStore(context: Context) {
    private val file = File(context.applicationContext.filesDir, "history.json")

    fun append(entry: HistoryEntry) {
        try {
            val arr = readArray()
            val obj = JSONObject().apply {
                put("ts", entry.ts)
                put("logDir", entry.logDir)
                val r = JSONObject()
                for ((k, v) in entry.results) r.put(k, v.toDouble())
                put("results", r)
            }
            arr.put(obj)
            writeArrayAtomic(arr)
        } catch (e: Exception) {
            Log.w(TAG, "history append failed", e)
        }
    }

    fun loadAll(): List<HistoryEntry> {
        return try {
            val arr = readArray()
            (0 until arr.length()).mapNotNull { i ->
                val o = arr.optJSONObject(i) ?: return@mapNotNull null
                val ts = o.optLong("ts", 0L)
                val logDir = o.optString("logDir", "")
                val r = o.optJSONObject("results") ?: JSONObject()
                val results = mutableMapOf<String, Float>()
                val it = r.keys()
                while (it.hasNext()) {
                    val k = it.next()
                    val v = r.optDouble(k, Double.NaN).toFloat()
                    if (!v.isNaN()) results[k] = v
                }
                HistoryEntry(ts, results.toMap(), logDir)
            }.sortedBy { it.ts }
        } catch (e: Exception) {
            Log.w(TAG, "history load failed", e)
            emptyList()
        }
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
            // Cross-device renames or stale targets — fall back to delete + rename.
            file.delete()
            if (!tmp.renameTo(file)) tmp.copyTo(file, overwrite = true)
        }
    }

    companion object { const val TAG = "HistoryStore" }
}
