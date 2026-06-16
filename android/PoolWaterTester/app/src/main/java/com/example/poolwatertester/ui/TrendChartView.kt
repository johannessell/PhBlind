package com.example.poolwatertester.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import com.example.poolwatertester.data.TargetRange
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.hypot

/** Single-series line chart with a shaded target band, custom marker shape,
 *  and tap-to-point. Hand-rolled on Canvas because pulling in MPAndroidChart
 *  needs JitPack network access which the local SSL setup blocks. Kept
 *  intentionally minimal: one series per chart, fixed colour + marker for
 *  colour-blind contrast, no zoom/pan. */
class TrendChartView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    init {
        // Without this, View base class ignores ACTION_DOWN, so we never get
        // ACTION_UP back here and the tap-to-point callback can't fire.
        isClickable = true
        isFocusable = true
    }

    data class Point(val ts: Long, val y: Float)
    enum class MarkerShape { CIRCLE, SQUARE, TRIANGLE }

    private var title: String = ""
    private var points: List<Point> = emptyList()
    private var range: TargetRange? = null
    private var seriesColor: Int = Color.parseColor("#1976D2")
    private var markerShape: MarkerShape = MarkerShape.CIRCLE

    private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = sp(14f); color = Color.DKGRAY
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = sp(10f); color = Color.GRAY
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#22000000")
        strokeWidth = dp(1f)
    }
    private val bandPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#3340B26B")
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = dp(2f); isDither = true
    }
    private val markerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val markerStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = dp(1.5f); color = Color.WHITE
    }
    private val emptyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = sp(12f); color = Color.GRAY
        textAlign = Paint.Align.CENTER
    }

    var onPointTap: ((Long) -> Unit)? = null

    private val plotArea = RectF()
    private val pxPositions = ArrayList<PointF2>()

    fun setData(
        title: String,
        points: List<Point>,
        range: TargetRange?,
        seriesColor: Int,
        markerShape: MarkerShape,
    ) {
        this.title = title
        this.points = points.sortedBy { it.ts }
        this.range = range
        this.seriesColor = seriesColor
        this.markerShape = markerShape
        linePaint.color = seriesColor
        markerPaint.color = seriesColor
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        val h = dp(180f).toInt()
        setMeasuredDimension(w, h)
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        canvas.drawText(title, dp(12f), dp(16f), titlePaint)

        // Plot area: leave room for Y labels on the left, X labels at bottom.
        val padL = dp(44f); val padR = dp(12f)
        val padT = dp(24f); val padB = dp(20f)
        plotArea.set(padL, padT, w - padR, h - padB)

        if (points.isEmpty()) {
            canvas.drawText("No data yet", w / 2f, plotArea.centerY(), emptyPaint)
            return
        }

        // Y domain: span data ± range, plus a small headroom.
        var yMin = points.minOf { it.y }
        var yMax = points.maxOf { it.y }
        range?.let { yMin = minOf(yMin, it.min); yMax = maxOf(yMax, it.max) }
        if (yMax - yMin < 0.01f) { yMin -= 0.5f; yMax += 0.5f }
        val pad = 0.10f * (yMax - yMin); yMin -= pad; yMax += pad

        // X domain: ts range. Single point → fake a 1-day span centred on it.
        val xMin = points.first().ts
        val xMax = points.last().ts
        val xSpan = (xMax - xMin).coerceAtLeast(86_400_000L) // 1 day
        val xMaxAdj = xMin + xSpan

        // Target band (translucent green).
        range?.let { r ->
            val yTop = yToPx(r.max, yMin, yMax)
            val yBot = yToPx(r.min, yMin, yMax)
            canvas.drawRect(plotArea.left, yTop, plotArea.right, yBot, bandPaint)
        }

        // Y ticks (3): min / mid / max of the auto domain.
        listOf(yMin, (yMin + yMax) * 0.5f, yMax).forEach { v ->
            val py = yToPx(v, yMin, yMax)
            canvas.drawLine(plotArea.left, py, plotArea.right, py, gridPaint)
            canvas.drawText(
                String.format(Locale.US, "%.2f", v),
                dp(4f), py + dp(4f), axisPaint
            )
        }

        // X labels: leftmost + rightmost dates.
        val fmt = SimpleDateFormat("MMM d", Locale.getDefault())
        canvas.drawText(fmt.format(Date(xMin)),
            plotArea.left, plotArea.bottom + dp(14f), axisPaint)
        canvas.drawText(fmt.format(Date(xMaxAdj)),
            plotArea.right - axisPaint.measureText(fmt.format(Date(xMaxAdj))),
            plotArea.bottom + dp(14f), axisPaint)

        // Line + markers.
        pxPositions.clear()
        val path = Path()
        points.forEachIndexed { i, p ->
            val px = xToPx(p.ts, xMin, xMaxAdj)
            val py = yToPx(p.y, yMin, yMax)
            pxPositions.add(PointF2(px, py, p.ts))
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        canvas.drawPath(path, linePaint)
        val r = dp(4.5f)
        pxPositions.forEach { pp ->
            drawMarker(canvas, pp.x, pp.y, r)
        }
    }

    private fun drawMarker(canvas: Canvas, cx: Float, cy: Float, r: Float) {
        when (markerShape) {
            MarkerShape.CIRCLE -> {
                canvas.drawCircle(cx, cy, r, markerPaint)
                canvas.drawCircle(cx, cy, r, markerStrokePaint)
            }
            MarkerShape.SQUARE -> {
                canvas.drawRect(cx - r, cy - r, cx + r, cy + r, markerPaint)
                canvas.drawRect(cx - r, cy - r, cx + r, cy + r, markerStrokePaint)
            }
            MarkerShape.TRIANGLE -> {
                val p = Path().apply {
                    moveTo(cx, cy - r * 1.15f)
                    lineTo(cx + r, cy + r * 0.7f)
                    lineTo(cx - r, cy + r * 0.7f); close()
                }
                canvas.drawPath(p, markerPaint)
                canvas.drawPath(p, markerStrokePaint)
            }
        }
    }

    private fun yToPx(v: Float, yMin: Float, yMax: Float): Float {
        val t = (v - yMin) / (yMax - yMin).coerceAtLeast(1e-6f)
        return plotArea.bottom - t * (plotArea.bottom - plotArea.top)
    }

    private fun xToPx(ts: Long, xMin: Long, xMax: Long): Float {
        val t = (ts - xMin).toFloat() / (xMax - xMin).coerceAtLeast(1L).toFloat()
        return plotArea.left + t * (plotArea.right - plotArea.left)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // Always let the base class handle ACTION_DOWN / cancel transitions
        // so pressed-state is set and ACTION_UP arrives back here. We only
        // *also* dispatch the tap callback on ACTION_UP.
        if (event.actionMasked == MotionEvent.ACTION_UP) {
            val cb = onPointTap
            if (cb != null) {
                // Generous tap radius — markers are 9dp across; 32dp covers a
                // typical finger pad without grabbing far-away points.
                val tapR = dp(32f)
                var best: PointF2? = null; var bestD = Float.MAX_VALUE
                for (pp in pxPositions) {
                    val d = hypot(event.x - pp.x, event.y - pp.y)
                    if (d < bestD) { bestD = d; best = pp }
                }
                if (best != null && bestD <= tapR) cb(best.ts)
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean = super.performClick()

    private fun dp(v: Float): Float = v * resources.displayMetrics.density
    private fun sp(v: Float): Float = v * resources.displayMetrics.scaledDensity

    private data class PointF2(val x: Float, val y: Float, val ts: Long)
}
