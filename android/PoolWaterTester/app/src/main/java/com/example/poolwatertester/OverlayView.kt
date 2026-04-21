package com.example.poolwatertester

import android.content.Context
import android.graphics.Canvas
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

    private var quad: FloatArray? = null
    private var frameW = 0
    private var frameH = 0
    private var progress = 0
    private var required = 1
    private var stable = false

    fun update(
        quad: FloatArray?,
        frameW: Int,
        frameH: Int,
        progress: Int,
        required: Int,
        stable: Boolean,
    ) {
        this.quad = quad
        this.frameW = frameW
        this.frameH = frameH
        this.progress = progress
        this.required = required.coerceAtLeast(1)
        this.stable = stable
        postInvalidate()
    }

    fun clear() {
        quad = null
        progress = 0
        stable = false
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val q = quad ?: return
        if (frameW == 0 || frameH == 0) return
        val viewW = width.toFloat()
        val viewH = height.toFloat()
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

        val barW = viewW * 0.5f
        val barH = 10f
        val barX = (viewW - barW) / 2f
        val barY = viewH - 120f
        canvas.drawRect(barX, barY, barX + barW, barY + barH, barBgPaint)
        val frac = progress.toFloat() / required.toFloat()
        canvas.drawRect(barX, barY, barX + barW * frac.coerceIn(0f, 1f), barY + barH, barFgPaint)
    }
}
