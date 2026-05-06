package com.example.poolwatertester

import android.content.Context
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val quadPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val barBgPaint = Paint().apply { color = 0x66000000 }
    private val barFgPaint = Paint().apply { color = 0xFF00C853.toInt() }
    // Always-on target guide: dashed rectangle showing the user where to
    // place the indicator and how large it should appear on screen.
    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = 0xCCFFFFFF.toInt()
        pathEffect = DashPathEffect(floatArrayOf(24f, 16f), 0f)
    }

    // Template aspect ratio (long/short side). Matches reference cards used
    // for calibration. setTargetAspect() lets callers override if a future
    // template has a different shape.
    private var targetAspect = 1.43f
    private val targetWidthFrac = 0.70f

    // EMA-smoothed quad we actually draw. Decoupled from the raw detection
    // so visual jitter doesn't track 30 fps detector noise.
    private var quad: FloatArray? = null
    private var frameW = 0
    private var frameH = 0
    private var progress = 0
    private var required = 1
    private var stable = false

    // alpha=0.35 -> ~3 frames of inertia, eye-pleasing without feeling laggy.
    private val smoothAlpha = 0.35f
    // Sub-pixel changes are invisible. Bigger threshold = fewer redraws but
    // overlay can briefly "snap" when accumulated change finally exceeds it.
    // 2 px is the sweet spot at typical phone DPI.
    private val redrawTolPx = 2.0f
    // Hold the last quad for this many frames of "no detection" before
    // clearing — single-frame detector drops shouldn't flicker the overlay.
    private val missGrace = 4
    private var missCount = 0

    fun update(
        quad: FloatArray?,
        frameW: Int,
        frameH: Int,
        progress: Int,
        required: Int,
        stable: Boolean,
    ) {
        val req = required.coerceAtLeast(1)

        // Miss-grace: a transient null doesn't blank the overlay or reset
        // progress; only after `missGrace` consecutive misses do we drop the
        // displayed quad. Without holding `progress` through the grace too,
        // the progress bar would oscillate 5→0→5 on every dropped frame.
        val held = quad == null && missCount < missGrace && this.quad != null
        val effectiveQuad: FloatArray?
        val effectiveProgress: Int
        val effectiveStable: Boolean
        if (quad == null) {
            missCount += 1
            if (held) {
                effectiveQuad = this.quad
                effectiveProgress = this.progress
                effectiveStable = this.stable
            } else {
                effectiveQuad = null
                effectiveProgress = 0
                effectiveStable = false
            }
        } else {
            missCount = 0
            effectiveQuad = quad
            effectiveProgress = progress
            effectiveStable = stable
        }

        val nextQuad = smoothQuad(this.quad, effectiveQuad)

        if (!hasChanged(nextQuad, frameW, frameH, effectiveProgress, req, effectiveStable)) {
            this.quad = nextQuad
            return
        }

        this.quad = nextQuad
        this.frameW = frameW
        this.frameH = frameH
        this.progress = effectiveProgress
        this.required = req
        this.stable = effectiveStable
        postInvalidate()
    }

    private fun smoothQuad(prev: FloatArray?, target: FloatArray?): FloatArray? {
        // Null transitions: no smoothing across them — would smear from old
        // position to new on re-acquisition.
        if (target == null) return null
        if (prev == null) return target.copyOf()
        val out = FloatArray(8)
        val a = smoothAlpha
        for (i in 0 until 8) out[i] = a * target[i] + (1 - a) * prev[i]
        return out
    }

    private fun hasChanged(
        newQuad: FloatArray?,
        newFrameW: Int,
        newFrameH: Int,
        newProgress: Int,
        newRequired: Int,
        newStable: Boolean,
    ): Boolean {
        if (newProgress != progress) return true
        if (newStable != stable) return true
        if (newRequired != required) return true
        if (newFrameW != frameW || newFrameH != frameH) return true
        val cur = quad
        if ((newQuad == null) != (cur == null)) return true
        if (newQuad == null || cur == null) return false
        for (i in 0 until 8) {
            if (kotlin.math.abs(newQuad[i] - cur[i]) > redrawTolPx) return true
        }
        return false
    }

    fun clear() {
        if (quad == null && progress == 0 && !stable) return
        quad = null
        progress = 0
        stable = false
        missCount = 0
        postInvalidate()
    }

    fun setTargetAspect(aspect: Float) {
        if (aspect > 0f && aspect != targetAspect) {
            targetAspect = aspect
            postInvalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val viewW = width.toFloat()
        val viewH = height.toFloat()

        // Always-on guide rectangle (drawn first so the live quad sits on top
        // of it). Aspect-matched to the template, ~70% of each view dimension
        // — orientation follows the view: portrait view => portrait guide.
        val frac = targetWidthFrac
        val guideW: Float
        val guideH: Float
        if (viewH >= viewW) {
            // Portrait: long side vertical. Short side ≤ frac*viewW; long
            // side ≤ frac*viewH; long = aspect*short.
            val short = minOf(viewW * frac, (viewH * frac) / targetAspect)
            guideW = short
            guideH = short * targetAspect
        } else {
            val short = minOf(viewH * frac, (viewW * frac) / targetAspect)
            guideH = short
            guideW = short * targetAspect
        }
        val gx = (viewW - guideW) / 2f
        val gy = (viewH - guideH) / 2f
        canvas.drawRect(gx, gy, gx + guideW, gy + guideH, guidePaint)

        val q = quad
        if (q != null && frameW != 0 && frameH != 0) {
            val scale = minOf(viewW / frameW, viewH / frameH)
            val ox = (viewW - frameW * scale) / 2f
            val oy = (viewH - frameH * scale) / 2f

            quadPaint.color = if (stable) 0xFF00C853.toInt() else 0xFFFFC107.toInt()
            val path = Path()
            path.moveTo(q[0] * scale + ox, q[1] * scale + oy)
            for (i in 1 until 4) {
                path.lineTo(q[i * 2] * scale + ox, q[i * 2 + 1] * scale + oy)
            }
            path.close()
            canvas.drawPath(path, quadPaint)
        }

        val barW = viewW * 0.5f
        val barH = 10f
        val barX = (viewW - barW) / 2f
        val barY = viewH - 120f
        canvas.drawRect(barX, barY, barX + barW, barY + barH, barBgPaint)
        val barFrac = progress.toFloat() / required.toFloat()
        canvas.drawRect(barX, barY, barX + barW * barFrac.coerceIn(0f, 1f), barY + barH, barFgPaint)
    }
}
