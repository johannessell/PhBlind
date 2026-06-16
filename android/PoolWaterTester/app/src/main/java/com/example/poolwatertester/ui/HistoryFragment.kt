package com.example.poolwatertester.ui

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.view.Gravity
import androidx.appcompat.app.AlertDialog
import androidx.core.widget.NestedScrollView
import androidx.fragment.app.Fragment
import com.example.poolwatertester.R
import com.example.poolwatertester.data.HistoryEntry
import com.example.poolwatertester.data.HistoryStore
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.data.Status
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.util.Locale
import com.google.android.material.button.MaterialButton
import com.google.android.material.snackbar.Snackbar

/** History view: one [TrendChartView] per parameter, stacked vertically.
 *  Each chart shows that parameter's values over time with the user's
 *  target range shaded in green. Tap a point → detail bottom sheet that
 *  can delete the entry. Clear-all lives in the header. */
class HistoryFragment : Fragment() {

    private lateinit var historyStore: HistoryStore
    private lateinit var settingsStore: SettingsStore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        historyStore = HistoryStore.forActive(requireContext())
        settingsStore = SettingsStore.forActive(requireContext())
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_history, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        view.findViewById<MaterialButton>(R.id.historyClearButton)
            .setOnClickListener { confirmClearAll() }
    }

    override fun onResume() {
        super.onResume()
        rebuild()
    }

    private fun rebuild() {
        val root = view ?: return
        val scroll = root.findViewById<NestedScrollView>(R.id.historyScroll)
        val header = root.findViewById<LinearLayout>(R.id.historyHeader)
        val filterScroll = root.findViewById<View>(R.id.paramFilterScroll)
        val holder = root.findViewById<LinearLayout>(R.id.chartsHolder)
        val placeholder = root.findViewById<TextView>(R.id.historyPlaceholder)
        val entries = historyStore.loadAll()

        if (entries.isEmpty()) {
            placeholder.visibility = View.VISIBLE
            scroll.visibility = View.GONE
            header.visibility = View.GONE
            filterScroll.visibility = View.GONE
            return
        }
        placeholder.visibility = View.GONE
        scroll.visibility = View.VISIBLE
        header.visibility = View.VISIBLE
        filterScroll.visibility = View.VISIBLE
        holder.removeAllViews()
        buildFilterChips(root.findViewById(R.id.paramFilterChips), entries)
        buildStats(root.findViewById(R.id.statsHolder), entries)

        val params = LinkedHashSet<String>().apply {
            addAll(SettingsStore.DEFAULT_RANGES.keys)
            for (e in entries) addAll(e.results.keys)
        }

        for (param in params) {
            // Honour the per-parameter "show in History" toggle from Settings.
            if (!settingsStore.isParamShown(param)) continue
            val points = entries.mapNotNull { e ->
                e.results[param]?.let { TrendChartView.Point(e.ts, it) }
            }
            if (points.isEmpty()) continue
            val chart = TrendChartView(requireContext()).apply {
                val (color, shape) = stylingFor(param)
                setData(param, points, settingsStore.rangeFor(param), color, shape)
                onPointTap = { ts ->
                    entries.firstOrNull { it.ts == ts }?.let { showDetail(it) }
                }
            }
            holder.addView(chart, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginStart = dp(12); marginEnd = dp(12)
                topMargin = dp(8); bottomMargin = dp(8)
            })
        }
    }

    private fun showDetail(entry: HistoryEntry) {
        HistoryDetailBottomSheet.newInstance(
            entry, settingsStore,
            onDeleted = {
                snack(getString(R.string.snack_deleted))
                rebuild()
            }
        ).show(parentFragmentManager, "history-detail")
    }

    private fun confirmClearAll() {
        AlertDialog.Builder(requireContext())
            .setTitle(R.string.confirm_clear_all_title)
            .setMessage(R.string.confirm_clear_all_message)
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(R.string.history_clear) { _, _ ->
                val n = historyStore.clearAll()
                val suffix = if (n == 1) getString(R.string.entry_singular)
                else getString(R.string.entry_plural)
                snack(getString(R.string.snack_cleared, n, suffix))
                rebuild()
            }
            .show()
    }

    private fun snack(msg: String) {
        val v = view ?: return
        Snackbar.make(v, msg, Snackbar.LENGTH_SHORT).show()
    }

    /** A row of filter chips, one per parameter, that toggle visibility
     *  of the corresponding chart and stats row. State persists in
     *  SettingsStore so the choice survives tab switches. */
    private fun buildFilterChips(group: ChipGroup, entries: List<HistoryEntry>) {
        group.removeAllViews()
        val params = LinkedHashSet<String>().apply {
            addAll(SettingsStore.DEFAULT_RANGES.keys)
            for (e in entries) addAll(e.results.keys)
        }
        for (param in params) {
            val chip = Chip(requireContext()).apply {
                text = param
                isCheckable = true
                isChecked = settingsStore.isParamShown(param)
                setOnCheckedChangeListener { _, on ->
                    settingsStore.setParamShown(param, on)
                    rebuild()
                }
            }
            group.addView(chip)
        }
    }

    /** One row per parameter: param name + mean (30d) + Δ7d signed +
     *  % of last-30-day samples that fell inside the configured range. */
    private fun buildStats(holder: LinearLayout, entries: List<HistoryEntry>) {
        holder.removeAllViews()
        val now = System.currentTimeMillis()
        val cutoff30 = now - 30L * 24 * 3_600_000
        val cutoff7  = now - 7L  * 24 * 3_600_000
        val params = LinkedHashSet<String>().apply {
            addAll(SettingsStore.DEFAULT_RANGES.keys)
            for (e in entries) addAll(e.results.keys)
        }
        val ctx = requireContext()
        for (param in params) {
            if (!settingsStore.isParamShown(param)) continue
            val recent = entries
                .filter { it.ts >= cutoff30 && it.results.containsKey(param) }
                .sortedBy { it.ts }
            if (recent.isEmpty()) continue
            val values = recent.map { it.results.getValue(param) }
            val mean = values.average().toFloat()
            val last = values.last()
            val ref7 = recent.lastOrNull { it.ts < cutoff7 }
                ?.results?.get(param)
            val delta = if (ref7 != null) last - ref7 else Float.NaN
            val range = settingsStore.rangeFor(param)
            val inRangePct = if (range != null) {
                val inN = values.count { range.classify(it) == Status.IN_RANGE }
                100f * inN / values.size
            } else Float.NaN
            holder.addView(statRow(ctx, param, mean, delta, inRangePct))
        }
    }

    private fun statRow(
        ctx: android.content.Context, param: String,
        mean: Float, delta: Float, inRangePct: Float,
    ): View {
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, dp(4))
        }
        val name = TextView(ctx).apply {
            text = param
            layoutParams = LinearLayout.LayoutParams(dp(56),
                LinearLayout.LayoutParams.WRAP_CONTENT)
        }
        val meanTv = TextView(ctx).apply {
            text = "${getString(R.string.stats_mean)} " +
                String.format(Locale.US, "%.2f", mean)
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val deltaStr = if (delta.isNaN()) "—"
        else String.format(Locale.US, "%+.2f", delta)
        val deltaTv = TextView(ctx).apply {
            text = "${getString(R.string.stats_delta)} $deltaStr"
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val pctStr = if (inRangePct.isNaN()) "—"
        else String.format(Locale.US, "%.0f%%", inRangePct)
        val pctTv = TextView(ctx).apply {
            text = "$pctStr ${getString(R.string.stats_in_range)}"
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        row.addView(name); row.addView(meanTv)
        row.addView(deltaTv); row.addView(pctTv)
        return row
    }

    /** Fixed colour + marker shape per parameter so they're always
     *  recognisable across charts and on a black-and-white printout. */
    private fun stylingFor(param: String): Pair<Int, TrendChartView.MarkerShape> {
        return when (param) {
            "pH"   -> Color.parseColor("#1976D2") to TrendChartView.MarkerShape.CIRCLE
            "H2O2" -> Color.parseColor("#D32F2F") to TrendChartView.MarkerShape.SQUARE
            "PHMB" -> Color.parseColor("#388E3C") to TrendChartView.MarkerShape.TRIANGLE
            else   -> Color.parseColor("#6D4C41") to TrendChartView.MarkerShape.CIRCLE
        }
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
