package com.example.poolwatertester.data

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** CSV serialisation for history entries. One row per parameter per entry
 *  (instead of one row per entry with N columns) so future parameters
 *  don't break the schema and partial imports still work. */
object HistoryCsv {

    private const val HEADER = "ts,iso_date,profile_id,profile_name,param,value,log_dir"
    private val ISO = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)

    fun toCsv(entries: List<HistoryEntry>, profileId: String, profileName: String): String {
        val sb = StringBuilder(HEADER).append('\n')
        for (e in entries.sortedBy { it.ts }) {
            val iso = ISO.format(Date(e.ts))
            for ((param, value) in e.results) {
                sb.append(e.ts).append(',')
                    .append(escape(iso)).append(',')
                    .append(escape(profileId)).append(',')
                    .append(escape(profileName)).append(',')
                    .append(escape(param)).append(',')
                    .append(value).append(',')
                    .append(escape(e.logDir)).append('\n')
            }
        }
        return sb.toString()
    }

    /** Parses rows back into [HistoryEntry], merging rows that share a ts.
     *  Unknown columns are ignored so newer-format files still load. */
    fun fromCsv(text: String): List<HistoryEntry> {
        val lines = text.split('\n').map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) return emptyList()
        val header = lines.first().split(',').map { it.trim() }
        val tsIdx = header.indexOf("ts")
        val paramIdx = header.indexOf("param")
        val valueIdx = header.indexOf("value")
        val logIdx = header.indexOf("log_dir")
        if (tsIdx < 0 || paramIdx < 0 || valueIdx < 0) return emptyList()

        val byTs = HashMap<Long, MutableMap<String, Float>>()
        val logByTs = HashMap<Long, String>()
        for (i in 1 until lines.size) {
            val row = split(lines[i])
            if (row.size <= valueIdx) continue
            val ts = row[tsIdx].toLongOrNull() ?: continue
            val param = row[paramIdx]
            val value = row[valueIdx].toFloatOrNull() ?: continue
            byTs.getOrPut(ts) { mutableMapOf() }[param] = value
            if (logIdx in row.indices && row[logIdx].isNotEmpty()) {
                logByTs.putIfAbsent(ts, row[logIdx])
            }
        }
        return byTs.entries.map { (ts, results) ->
            HistoryEntry(ts, results.toMap(),
                logByTs[ts] ?: "measurements/$ts")
        }
    }

    /** Splits a CSV row honouring double-quoted fields with embedded commas. */
    private fun split(line: String): List<String> {
        val out = ArrayList<String>()
        val cur = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val c = line[i]
            when {
                inQuotes && c == '"' && i + 1 < line.length && line[i + 1] == '"' -> {
                    cur.append('"'); i++
                }
                c == '"' -> inQuotes = !inQuotes
                c == ',' && !inQuotes -> { out.add(cur.toString()); cur.clear() }
                else -> cur.append(c)
            }
            i++
        }
        out.add(cur.toString())
        return out
    }

    private fun escape(s: String): String {
        return if (s.any { it == ',' || it == '"' || it == '\n' }) {
            "\"" + s.replace("\"", "\"\"") + "\""
        } else s
    }
}
