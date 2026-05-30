package com.example.poolwatertester.ui

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.widget.NestedScrollView
import androidx.fragment.app.Fragment
import com.example.poolwatertester.R
import com.example.poolwatertester.data.HistoryEntry
import com.example.poolwatertester.data.HistoryStore
import com.example.poolwatertester.data.SettingsStore

/** History view: one [TrendChartView] per parameter, stacked vertically.
 *  Each chart shows that parameter's values over time with the user's
 *  target range shaded in green. Taps on a point open the detail sheet
 *  with the original overlay image. */
class HistoryFragment : Fragment() {

    private lateinit var historyStore: HistoryStore
    private lateinit var settingsStore: SettingsStore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        historyStore = HistoryStore(requireContext())
        settingsStore = SettingsStore(requireContext())
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_history, container, false)

    override fun onResume() {
        super.onResume()
        rebuild()
    }

    /** Rebuilt on every onResume so a measurement taken on the Measure tab
     *  shows up immediately when the user comes back. */
    private fun rebuild() {
        val root = view ?: return
        val scroll = root.findViewById<NestedScrollView>(R.id.historyScroll)
        val holder = root.findViewById<LinearLayout>(R.id.chartsHolder)
        val placeholder = root.findViewById<TextView>(R.id.historyPlaceholder)
        val entries = historyStore.loadAll()

        if (entries.isEmpty()) {
            placeholder.visibility = View.VISIBLE
            scroll.visibility = View.GONE
            return
        }
        placeholder.visibility = View.GONE
        scroll.visibility = View.VISIBLE
        holder.removeAllViews()

        // Use the union of seeded params + anything seen in history so the
        // user sees a chart even for a parameter they've measured once.
        val params = LinkedHashSet<String>().apply {
            addAll(SettingsStore.DEFAULT_RANGES.keys)
            for (e in entries) addAll(e.results.keys)
        }

        for (param in params) {
            val points = entries.mapNotNull { e ->
                e.results[param]?.let { TrendChartView.Point(e.ts, it) }
            }
            if (points.isEmpty()) continue  // no datapoints for this param yet
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
        HistoryDetailBottomSheet.newInstance(entry, settingsStore)
            .show(parentFragmentManager, "history-detail")
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
