package com.example.poolwatertester

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.util.AttributeSet
import android.view.Gravity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.FrameLayout

/**
 * Overlay that hosts one EditText per cell + one per parameter header.
 * Children are laid out in absolute pixel coords derived from the cell
 * rect inside the reference bitmap, mapped through the fitCenter scale
 * of the backing ImageView (same size as this view).
 *
 * Callers set bitmap dimensions + data via [setReference]; user edits
 * are written back into the same mutable maps (cellValues, paramNames)
 * so the activity can serialize on Save.
 */
class CellOverlayEditor @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : FrameLayout(context, attrs) {

    data class Cell(
        val cellIdx: Int,
        val rowIdx: Int,
        val groupIdx: Int?,
        val isColor: Boolean,
        val x: Int, val y: Int, val w: Int, val h: Int,
    )

    data class ParamHeader(
        val groupIdx: Int,
        val xCenter: Int,
        val yTop: Int,  // placed above the first cell row
        val width: Int,
    )

    private var bmpW = 0
    private var bmpH = 0
    private var cells: List<Cell> = emptyList()
    private var headers: List<ParamHeader> = emptyList()

    private val cellEdits = HashMap<Int, EditText>()
    private val headerEdits = HashMap<Int, EditText>()

    val cellValues = HashMap<Int, String>()
    val paramNames = HashMap<Int, String>()

    fun setReference(
        bitmapWidth: Int,
        bitmapHeight: Int,
        cells: List<Cell>,
        headers: List<ParamHeader>,
        initialValues: Map<Int, String?>,
        initialNames: Map<Int, String?>,
    ) {
        bmpW = bitmapWidth
        bmpH = bitmapHeight
        this.cells = cells
        this.headers = headers
        cellValues.clear()
        paramNames.clear()
        cellEdits.clear()
        headerEdits.clear()
        removeAllViews()

        for (c in cells) {
            if (c.isColor) continue  // swatches get value from their row's measure cell
            val et = makeEdit(isNumber = true).apply {
                setText(initialValues[c.cellIdx] ?: "")
                addTextChangedListener(object : android.text.TextWatcher {
                    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                    override fun afterTextChanged(s: android.text.Editable?) {
                        cellValues[c.cellIdx] = s?.toString() ?: ""
                    }
                })
            }
            cellEdits[c.cellIdx] = et
            addView(et)
            cellValues[c.cellIdx] = initialValues[c.cellIdx] ?: ""
        }

        for (h in headers) {
            val et = makeEdit(isNumber = false).apply {
                setText(initialNames[h.groupIdx] ?: "")
                addTextChangedListener(object : android.text.TextWatcher {
                    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                    override fun afterTextChanged(s: android.text.Editable?) {
                        paramNames[h.groupIdx] = s?.toString() ?: ""
                    }
                })
            }
            headerEdits[h.groupIdx] = et
            addView(et)
            paramNames[h.groupIdx] = initialNames[h.groupIdx] ?: ""
        }

        requestLayout()
    }

    private fun makeEdit(isNumber: Boolean): EditText = EditText(context).apply {
        gravity = Gravity.CENTER
        setPadding(4, 2, 4, 2)
        setTextColor(Color.WHITE)
        setHintTextColor(0x99FFFFFF.toInt())
        textSize = 12f
        inputType = if (isNumber)
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
        else
            InputType.TYPE_CLASS_TEXT
        imeOptions = EditorInfo.IME_ACTION_DONE
        setSingleLine(true)
        hint = "—"
        background = GradientDrawable().apply {
            setColor(0x66000000)
            setStroke(2, 0xFFFFFFFF.toInt())
            cornerRadius = 4f
        }
        setOnFocusChangeListener { v, hasFocus ->
            val bg = GradientDrawable().apply {
                setColor(if (hasFocus) 0xCC000000.toInt() else 0x66000000)
                setStroke(2, if (hasFocus) 0xFFFFC107.toInt() else 0xFFFFFFFF.toInt())
                cornerRadius = 4f
            }
            v.background = bg
        }
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        if (bmpW <= 0 || bmpH <= 0) return
        val viewW = (right - left).toFloat()
        val viewH = (bottom - top).toFloat()
        val scale = minOf(viewW / bmpW, viewH / bmpH)
        val imgW = bmpW * scale
        val imgH = bmpH * scale
        val ox = (viewW - imgW) * 0.5f
        val oy = (viewH - imgH) * 0.5f

        for (c in cells) {
            if (c.isColor) continue
            val et = cellEdits[c.cellIdx] ?: continue
            val l = (c.x * scale + ox).toInt()
            val t = (c.y * scale + oy).toInt()
            val r = ((c.x + c.w) * scale + ox).toInt()
            val b = ((c.y + c.h) * scale + oy).toInt()
            et.layout(l, t, r, b)
        }

        val headerH = (bmpH * scale * 0.05f).coerceAtLeast(32f).toInt()
        for (h in headers) {
            val et = headerEdits[h.groupIdx] ?: continue
            val cx = h.xCenter * scale + ox
            val half = h.width * scale * 0.5f
            val l = (cx - half).toInt()
            val r = (cx + half).toInt()
            val t = (h.yTop * scale + oy - headerH - 4f).toInt().coerceAtLeast(0)
            val b = t + headerH
            et.layout(l, t, r, b)
        }
    }

    override fun generateDefaultLayoutParams(): LayoutParams =
        LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
}
