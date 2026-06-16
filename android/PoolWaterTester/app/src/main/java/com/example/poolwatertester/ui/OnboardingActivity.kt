package com.example.poolwatertester.ui

import android.app.TimePickerDialog
import android.content.Intent
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
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.RecyclerView
import androidx.viewpager2.widget.ViewPager2
import com.example.poolwatertester.MainActivity
import com.example.poolwatertester.R
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.data.TargetRange
import com.example.poolwatertester.reminder.ReminderReceiver
import com.example.poolwatertester.reminder.ReminderScheduler
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import java.util.Locale

/** 3-page first-run pager: explain the guide rect, set target ranges,
 *  optionally enable the reminder. On Done we set `onboardingDone` and
 *  jump to MainActivity. Swipe-back inside the pager works through
 *  ViewPager2; the bottom Skip/Next buttons mirror that. */
class OnboardingActivity : AppCompatActivity() {

    private lateinit var store: SettingsStore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_onboarding)
        store = SettingsStore.forActive(this)
        store.seedDefaultsIfMissing(SettingsStore.DEFAULT_RANGES.keys)
        ReminderReceiver.ensureChannel(this)

        val pager = findViewById<ViewPager2>(R.id.pager)
        val skip = findViewById<MaterialButton>(R.id.skipButton)
        val next = findViewById<MaterialButton>(R.id.nextButton)
        pager.adapter = PageAdapter(store)

        fun updateNextLabel() {
            next.text = if (pager.currentItem == PAGE_COUNT - 1)
                getString(R.string.onboarding_done)
            else getString(R.string.onboarding_next)
        }
        updateNextLabel()
        pager.registerOnPageChangeCallback(
            object : ViewPager2.OnPageChangeCallback() {
                override fun onPageSelected(position: Int) { updateNextLabel() }
            }
        )

        skip.setOnClickListener { finishOnboarding() }
        next.setOnClickListener {
            if (pager.currentItem < PAGE_COUNT - 1) pager.currentItem++
            else finishOnboarding()
        }
    }

    private fun finishOnboarding() {
        store.onboardingDone = true
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }

    // ----------------------------------------------------------- adapter

    private class PageAdapter(val store: SettingsStore)
        : RecyclerView.Adapter<PageVh>() {

        override fun getItemCount() = PAGE_COUNT

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PageVh {
            val v = LayoutInflater.from(parent.context)
                .inflate(R.layout.page_onboarding, parent, false)
            v.layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            return PageVh(v)
        }

        override fun onBindViewHolder(holder: PageVh, position: Int) {
            holder.bind(position, store)
        }
    }

    private class PageVh(view: View) : RecyclerView.ViewHolder(view) {
        private val title: TextView = view.findViewById(R.id.pageTitle)
        private val body: TextView = view.findViewById(R.id.pageBody)
        private val extras: ViewGroup = view.findViewById(R.id.pageExtras)
        private val ctx = view.context

        fun bind(position: Int, store: SettingsStore) {
            extras.removeAllViews()
            when (position) {
                0 -> {
                    title.text = ctx.getString(R.string.onboarding_page1_title)
                    body.text = ctx.getString(R.string.onboarding_page1_body)
                }
                1 -> {
                    title.text = ctx.getString(R.string.onboarding_page2_title)
                    body.text = ctx.getString(R.string.onboarding_page2_body)
                    extras.addView(rangesBlock(store))
                }
                2 -> {
                    title.text = ctx.getString(R.string.onboarding_page3_title)
                    body.text = ctx.getString(R.string.onboarding_page3_body)
                    extras.addView(reminderBlock(store))
                }
            }
        }

        private fun dp(v: Int): Int =
            (v * ctx.resources.displayMetrics.density).toInt()

        private fun rangesBlock(store: SettingsStore): View {
            val container = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
            }
            for (param in SettingsStore.DEFAULT_RANGES.keys) {
                container.addView(rangeRow(param, store))
            }
            return container
        }

        private fun rangeRow(param: String, store: SettingsStore): View {
            val row = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(4), 0, dp(4))
            }
            val label = TextView(ctx).apply {
                text = param; textSize = 16f
                layoutParams = LinearLayout.LayoutParams(dp(56),
                    LinearLayout.LayoutParams.WRAP_CONTENT)
            }
            row.addView(label)
            val current = store.rangeFor(param)
                ?: SettingsStore.DEFAULT_RANGES[param]!!
            val minE = floatEdit(current.min)
            val idE = floatEdit(current.ideal)
            val mxE = floatEdit(current.max)
            row.addView(labelled(ctx.getString(R.string.range_min), minE))
            row.addView(labelled(ctx.getString(R.string.range_ideal), idE))
            row.addView(labelled(ctx.getString(R.string.range_max), mxE))
            val watcher = object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(s: Editable?) {
                    val a = minE.text.toString().toFloatOrNull()
                    val b = idE.text.toString().toFloatOrNull()
                    val c = mxE.text.toString().toFloatOrNull()
                    if (a != null && b != null && c != null && a <= c) {
                        store.setRange(param, TargetRange(a, b, c))
                    }
                }
            }
            minE.addTextChangedListener(watcher)
            idE.addTextChangedListener(watcher)
            mxE.addTextChangedListener(watcher)
            return row
        }

        private fun labelled(cap: String, edit: EditText): View {
            val col = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginStart = dp(4); marginEnd = dp(4)
                }
            }
            col.addView(TextView(ctx).apply { text = cap; textSize = 11f })
            col.addView(edit)
            return col
        }

        private fun floatEdit(initial: Float): EditText {
            return EditText(ctx).apply {
                setText(String.format(Locale.US, "%.2f", initial))
                inputType = InputType.TYPE_CLASS_NUMBER or
                    InputType.TYPE_NUMBER_FLAG_DECIMAL
                gravity = Gravity.END
                setSingleLine(true)
            }
        }

        private fun reminderBlock(store: SettingsStore): View {
            val container = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
            }
            val sw = MaterialSwitch(ctx).apply {
                text = ctx.getString(R.string.settings_reminder_switch)
                isChecked = store.reminderEnabled
            }
            val daysRow = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(8), 0, dp(8))
            }
            daysRow.addView(TextView(ctx).apply {
                text = ctx.getString(R.string.settings_reminder_days_label)
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
            val daysE = EditText(ctx).apply {
                inputType = InputType.TYPE_CLASS_NUMBER
                setText(store.reminderIntervalDays.toString())
                layoutParams = LinearLayout.LayoutParams(dp(80),
                    LinearLayout.LayoutParams.WRAP_CONTENT)
            }
            daysRow.addView(daysE)
            val timeRow = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(8), 0, dp(8))
            }
            timeRow.addView(TextView(ctx).apply {
                text = ctx.getString(R.string.settings_reminder_time_label)
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
            val timeBtn = MaterialButton(ctx).apply {
                text = String.format(Locale.getDefault(),
                    "%02d:%02d", store.reminderHour, store.reminderMinute)
            }
            timeRow.addView(timeBtn)
            container.addView(sw); container.addView(daysRow); container.addView(timeRow)

            fun apply() {
                store.reminderEnabled = sw.isChecked
                store.reminderIntervalDays = daysE.text.toString()
                    .toIntOrNull()?.coerceAtLeast(1) ?: 1
                if (store.reminderEnabled) {
                    ReminderScheduler.enable(ctx,
                        store.reminderIntervalDays,
                        store.reminderHour, store.reminderMinute)
                } else {
                    ReminderScheduler.disable(ctx)
                }
            }
            sw.setOnCheckedChangeListener { _, _ -> apply() }
            daysE.addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(s: Editable?) { apply() }
            })
            timeBtn.setOnClickListener {
                TimePickerDialog(ctx, { _, h, m ->
                    store.reminderHour = h
                    store.reminderMinute = m
                    timeBtn.text = String.format(Locale.getDefault(),
                        "%02d:%02d", h, m)
                    apply()
                }, store.reminderHour, store.reminderMinute, true).show()
            }
            return container
        }
    }

    companion object { private const val PAGE_COUNT = 3 }
}
