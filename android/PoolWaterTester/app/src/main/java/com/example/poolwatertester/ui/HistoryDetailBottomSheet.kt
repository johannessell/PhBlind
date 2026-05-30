package com.example.poolwatertester.ui

import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.example.poolwatertester.R
import com.example.poolwatertester.data.HistoryEntry
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.data.Status
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Detail of a single past measurement: timestamp, the overlay image we
 *  saved next to the input frame, and per-parameter values with the same
 *  colour-coded status pills as the live result card. */
class HistoryDetailBottomSheet : BottomSheetDialogFragment() {

    private var entry: HistoryEntry? = null
    private var settingsStore: SettingsStore? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.dialog_history_detail, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val e = entry ?: return run { dismiss(); Unit }
        val store = settingsStore ?: SettingsStore(requireContext())

        val title = view.findViewById<TextView>(R.id.detailTitle)
        val overlay = view.findViewById<ImageView>(R.id.detailOverlay)
        val rows = view.findViewById<LinearLayout>(R.id.detailRows)

        title.text = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            .format(Date(e.ts))

        // overlay image lives next to the input frame
        val ctx = requireContext()
        val overlayFile = File(ctx.getExternalFilesDir(null), "${e.logDir}/overlay.jpg")
        if (overlayFile.isFile) {
            val bmp = BitmapFactory.decodeFile(overlayFile.absolutePath)
            if (bmp != null) {
                overlay.setImageBitmap(bmp)
                overlay.visibility = View.VISIBLE
            }
        }

        rows.removeAllViews()
        for ((param, value) in e.results) {
            val range = store.rangeFor(param)
            val status = range?.classify(value) ?: Status.UNKNOWN
            rows.addView(buildRow(param, value, status))
        }
    }

    private fun buildRow(param: String, value: Float, status: Status): View {
        val ctx = requireContext()
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(6); bottomMargin = dp(6) }
        }
        val label = TextView(ctx).apply {
            text = String.format("%-5s  %.2f", param, value)
            textSize = 16f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
        }
        val (symbol, bg) = when (status) {
            Status.IN_RANGE -> "✓ OK"   to R.drawable.pill_in_range
            Status.NEAR     -> "⚠ NEAR" to R.drawable.pill_near
            Status.OUT      -> "✕ OUT"  to R.drawable.pill_out
            Status.UNKNOWN  -> "?"      to R.drawable.pill_unknown
        }
        val pill = TextView(ctx).apply {
            text = symbol
            setTextColor(Color.WHITE)
            textSize = 12f
            setBackgroundResource(bg)
            gravity = Gravity.CENTER
        }
        row.addView(label)
        row.addView(pill)
        return row
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    companion object {
        fun newInstance(entry: HistoryEntry, settings: SettingsStore): HistoryDetailBottomSheet {
            return HistoryDetailBottomSheet().also {
                it.entry = entry
                it.settingsStore = settings
            }
        }
    }
}
