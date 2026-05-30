package com.example.poolwatertester.ui

import android.Manifest
import android.app.TimePickerDialog
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.poolwatertester.R
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.data.TargetRange
import com.example.poolwatertester.reminder.ReminderReceiver
import com.example.poolwatertester.reminder.ReminderScheduler
import com.google.android.material.materialswitch.MaterialSwitch
import java.util.Locale

/** Target-range editors + reminder controls. Every edit auto-persists to
 *  [SettingsStore]; the reminder switch + interval + time also reschedule
 *  the alarm so the next firing matches the new config. */
class SettingsFragment : Fragment() {

    private lateinit var store: SettingsStore
    private var reminderSwitchRef: MaterialSwitch? = null
    private var reminderStatusRef: TextView? = null

    private val requestNotifPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            // Permission accepted — proceed with scheduling.
            scheduleNow()
        } else {
            // Revert the switch; persisted state stays "off".
            reminderSwitchRef?.isChecked = false
            store.reminderEnabled = false
            reminderStatusRef?.text = "Notifications denied — reminders disabled"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = SettingsStore(requireContext())
        store.seedDefaultsIfMissing(SettingsStore.DEFAULT_RANGES.keys)
        ReminderReceiver.ensureChannel(requireContext())
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_settings, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        buildRangeRows(view.findViewById(R.id.rangesHolder))
        wireReminder(view)
    }

    // -------------------------------------------------------------- ranges

    private fun buildRangeRows(holder: LinearLayout) {
        holder.removeAllViews()
        for (param in SettingsStore.DEFAULT_RANGES.keys) {
            holder.addView(rangeRow(param))
        }
    }

    private fun rangeRow(param: String): View {
        val ctx = requireContext()
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(4); bottomMargin = dp(4) }
        }
        val label = TextView(ctx).apply {
            text = param
            textSize = 16f
            layoutParams = LinearLayout.LayoutParams(dp(56),
                LinearLayout.LayoutParams.WRAP_CONTENT)
        }
        row.addView(label)
        val current = store.rangeFor(param) ?: SettingsStore.DEFAULT_RANGES[param]!!
        val minEdit = makeFloatEdit(current.min)
        val idealEdit = makeFloatEdit(current.ideal)
        val maxEdit = makeFloatEdit(current.max)
        row.addView(labelled(ctx, "min", minEdit))
        row.addView(labelled(ctx, "ideal", idealEdit))
        row.addView(labelled(ctx, "max", maxEdit))

        val watcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val mn = minEdit.text.toString().toFloatOrNull()
                val id = idealEdit.text.toString().toFloatOrNull()
                val mx = maxEdit.text.toString().toFloatOrNull()
                if (mn != null && mx != null && id != null && mn <= mx) {
                    store.setRange(param, TargetRange(mn, id, mx))
                }
            }
        }
        minEdit.addTextChangedListener(watcher)
        idealEdit.addTextChangedListener(watcher)
        maxEdit.addTextChangedListener(watcher)
        return row
    }

    private fun labelled(ctx: android.content.Context, caption: String, edit: EditText): View {
        val col = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = dp(4); marginEnd = dp(4)
            }
        }
        col.addView(TextView(ctx).apply {
            text = caption; textSize = 11f
        })
        col.addView(edit)
        return col
    }

    private fun makeFloatEdit(initial: Float): EditText {
        return EditText(requireContext()).apply {
            setText(String.format(Locale.US, "%.2f", initial))
            inputType = InputType.TYPE_CLASS_NUMBER or
                    InputType.TYPE_NUMBER_FLAG_DECIMAL
            gravity = Gravity.END
            setSingleLine(true)
        }
    }

    // ------------------------------------------------------------- reminder

    private fun wireReminder(view: View) {
        val sw = view.findViewById<MaterialSwitch>(R.id.reminderSwitch)
        val days = view.findViewById<EditText>(R.id.reminderDays)
        val timeBtn = view.findViewById<android.widget.Button>(R.id.reminderTimeButton)
        val status = view.findViewById<TextView>(R.id.reminderStatus)

        reminderSwitchRef = sw
        reminderStatusRef = status

        sw.isChecked = store.reminderEnabled
        days.setText(store.reminderIntervalDays.toString())
        timeBtn.text = String.format(Locale.getDefault(),
            "%02d:%02d", store.reminderHour, store.reminderMinute)

        fun applySettings(reschedule: Boolean) {
            val enabled = sw.isChecked
            val n = days.text.toString().toIntOrNull()?.coerceAtLeast(1) ?: 1
            store.reminderEnabled = enabled
            store.reminderIntervalDays = n
            if (!enabled) {
                ReminderScheduler.disable(requireContext())
                status.text = "Reminders off"
                return
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                ContextCompat.checkSelfPermission(
                    requireContext(), Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                // We need the runtime permission before any notification will
                // actually appear. Request it; the callback re-runs the
                // scheduling on grant, or flips the switch back on deny.
                requestNotifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                status.text = "Asking for notification permission…"
                return
            }
            if (reschedule) scheduleNow()
        }

        sw.setOnCheckedChangeListener { _, _ -> applySettings(true) }
        days.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) { applySettings(true) }
        })
        timeBtn.setOnClickListener {
            TimePickerDialog(requireContext(), { _, h, m ->
                store.reminderHour = h
                store.reminderMinute = m
                timeBtn.text = String.format(Locale.getDefault(), "%02d:%02d", h, m)
                applySettings(true)
            }, store.reminderHour, store.reminderMinute, true).show()
        }
        applySettings(false)  // refresh status line without re-arming
    }

    /** Push the currently-persisted values to AlarmManager. Called when
     *  scheduling is safe (permission granted or pre-Tiramisu). */
    private fun scheduleNow() {
        ReminderScheduler.enable(requireContext(),
            store.reminderIntervalDays, store.reminderHour, store.reminderMinute)
        reminderStatusRef?.text = "Next reminder approx every " +
            "${store.reminderIntervalDays} day(s) at " +
            String.format(Locale.getDefault(), "%02d:%02d",
                store.reminderHour, store.reminderMinute)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        reminderSwitchRef = null
        reminderStatusRef = null
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
