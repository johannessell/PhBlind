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
        val colIdx: Int,
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
    private val colorOutlines = HashMap<Int, View>()
    private var initialValues: Map<Int, String?> = emptyMap()
    private var initialNames: Map<Int, String?> = emptyMap()

    val cellValues = HashMap<Int, String>()
    val paramNames = HashMap<Int, String>()
    /** Per-column current type (true = color, false = measure). Starts
     *  from the cell's initial isColor, mutated by long-press toggle.
     *  Read by the activity on save and forwarded to Python. */
    val colIsColor = HashMap<Int, Boolean>()

    private fun groupColorFor(groupIdx: Int?): Int {
        if (groupIdx == null || groupIdx < 0) return 0xFFAAAAAA.toInt()
        val palette = intArrayOf(
            0xFFFFC107.toInt(),  // amber
            0xFF03A9F4.toInt(),  // light blue
            0xFF8BC34A.toInt(),  // light green
            0xFFE91E63.toInt(),  // pink
            0xFFFF5722.toInt(),  // deep orange
            0xFF9C27B0.toInt(),  // purple
        )
        return palette[groupIdx % palette.size]
    }

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
        this.initialValues = initialValues
        this.initialNames = initialNames
        cellValues.clear()
        paramNames.clear()
        // Per-column current type starts from the cell's classifier output.
        colIsColor.clear()
        for (c in cells) {
            // If a column has mixed cells (shouldn't happen) the LAST
            // wins — but in practice every cell in a column shares one
            // is_color_cell value.
            colIsColor[c.colIdx] = c.isColor
        }
        rebuildViews()
    }

    private fun rebuildViews() {
        cellEdits.clear()
        headerEdits.clear()
        colorOutlines.clear()
        removeAllViews()

        for (c in cells) {
            val groupColor = groupColorFor(c.groupIdx)
            val isColor = colIsColor[c.colIdx] ?: c.isColor
            if (isColor) {
                // Non-editable visual outline so the parameter grouping
                // (measure + its color swatches share a colour) is
                // visible. No input field — value comes from the row's
                // measure cell. Long-press toggles the column to
                // 'measure' so the user can type a value.
                val v = View(context).apply {
                    isLongClickable = true
                    background = GradientDrawable().apply {
                        setColor(0x22000000)
                        setStroke(3, groupColor)
                        cornerRadius = 4f
                    }
                    setOnLongClickListener {
                        toggleColumnType(c.colIdx)
                        true
                    }
                }
                colorOutlines[c.cellIdx] = v
                addView(v)
                continue
            }
            val et = makeEdit(isNumber = true, strokeColor = groupColor).apply {
                setText(cellValues[c.cellIdx] ?: initialValues[c.cellIdx] ?: "")
                addTextChangedListener(object : android.text.TextWatcher {
                    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                    override fun afterTextChanged(s: android.text.Editable?) {
                        cellValues[c.cellIdx] = s?.toString() ?: ""
                    }
                })
                setOnLongClickListener {
                    toggleColumnType(c.colIdx)
                    true
                }
            }
            cellEdits[c.cellIdx] = et
            addView(et)
            if (c.cellIdx !in cellValues) {
                cellValues[c.cellIdx] = initialValues[c.cellIdx] ?: ""
            }
        }

        for (h in headers) {
            val et = makeEdit(isNumber = false,
                              strokeColor = groupColorFor(h.groupIdx)).apply {
                setText(paramNames[h.groupIdx] ?: initialNames[h.groupIdx] ?: "")
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
            if (h.groupIdx !in paramNames) {
                paramNames[h.groupIdx] = initialNames[h.groupIdx] ?: ""
            }
        }

        requestLayout()
    }

    private fun toggleColumnType(colIdx: Int) {
        val cur = colIsColor[colIdx] ?: return
        colIsColor[colIdx] = !cur
        rebuildViews()
    }

    private fun makeEdit(isNumber: Boolean,
                         strokeColor: Int = 0xFFFFFFFF.toInt()): EditText =
        EditText(context).apply {
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
                setStroke(2, strokeColor)
                cornerRadius = 4f
            }
            setOnFocusChangeListener { v, hasFocus ->
                val bg = GradientDrawable().apply {
                    setColor(if (hasFocus) 0xCC000000.toInt() else 0x66000000)
                    setStroke(if (hasFocus) 3 else 2,
                              if (hasFocus) 0xFFFFFFFF.toInt() else strokeColor)
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
            val l = (c.x * scale + ox).toInt()
            val t = (c.y * scale + oy).toInt()
            val r = ((c.x + c.w) * scale + ox).toInt()
            val b = ((c.y + c.h) * scale + oy).toInt()
            // Use the CURRENT column type (possibly toggled by the user
            // via long-press), not the frozen classifier output on the
            // Cell record, so the freshly-rebuilt view actually gets a
            // bounding rect.
            val isColor = colIsColor[c.colIdx] ?: c.isColor
            if (isColor) {
                colorOutlines[c.cellIdx]?.layout(l, t, r, b)
            } else {
                cellEdits[c.cellIdx]?.layout(l, t, r, b)
            }
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
