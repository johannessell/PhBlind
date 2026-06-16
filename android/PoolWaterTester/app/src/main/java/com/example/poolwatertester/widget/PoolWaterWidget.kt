package com.example.poolwatertester.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.text.format.DateUtils
import android.widget.RemoteViews
import com.example.poolwatertester.MainActivity
import com.example.poolwatertester.R
import com.example.poolwatertester.data.HistoryEntry
import com.example.poolwatertester.data.HistoryStore
import com.example.poolwatertester.data.ProfilesStore
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.data.Status
import com.example.poolwatertester.data.TargetRange
import java.util.Calendar
import java.util.Locale

/** Home-screen widget showing the latest saved reading for the active
 *  pool + when the next reminder will fire. Tap → opens MainActivity. */
class PoolWaterWidget : AppWidgetProvider() {

    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray,
    ) {
        for (id in appWidgetIds) renderInto(context, appWidgetManager, id)
    }

    companion object {

        /** Called whenever the data backing the widget changes (after Save,
         *  after reminder change, on boot). Pushes a fresh RemoteViews to
         *  every installed instance. */
        fun refreshAll(context: Context) {
            val mgr = AppWidgetManager.getInstance(context)
            val ids = mgr.getAppWidgetIds(
                ComponentName(context, PoolWaterWidget::class.java))
            for (id in ids) renderInto(context, mgr, id)
        }

        private fun renderInto(
            context: Context, mgr: AppWidgetManager, id: Int,
        ) {
            val views = RemoteViews(context.packageName, R.layout.widget_pool_water)
            val profile = ProfilesStore(context).active()
            val settings = SettingsStore(context, profile.id)
            val history = HistoryStore(context, profile.id)
            val last = history.loadAll().lastOrNull()
            renderTop(context, views, profile.name, last)
            renderPills(views, last, settings)
            renderBottom(context, views, settings)

            val tap = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                        Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pi = PendingIntent.getActivity(context, 0, tap,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
            views.setOnClickPendingIntent(R.id.widgetRoot, pi)

            mgr.updateAppWidget(id, views)
        }

        private fun renderTop(
            context: Context, views: RemoteViews,
            profileName: String, last: HistoryEntry?,
        ) {
            val rel = if (last != null) DateUtils.getRelativeTimeSpanString(
                last.ts, System.currentTimeMillis(),
                DateUtils.MINUTE_IN_MILLIS
            ).toString() else context.getString(R.string.widget_empty)
            views.setTextViewText(R.id.widgetTopLine, "$profileName  •  $rel")
        }

        private fun renderPills(
            views: RemoteViews, last: HistoryEntry?, settings: SettingsStore,
        ) {
            val pillIds = intArrayOf(R.id.pill1, R.id.pill2, R.id.pill3)
            if (last == null) {
                for (id in pillIds) views.setViewVisibility(id, android.view.View.GONE)
                return
            }
            val entries = last.results.entries.toList()
            for ((i, id) in pillIds.withIndex()) {
                if (i >= entries.size) {
                    views.setViewVisibility(id, android.view.View.GONE); continue
                }
                val (param, value) = entries[i]
                val range = settings.rangeFor(param)
                val bg = bgFor(range, value)
                views.setViewVisibility(id, android.view.View.VISIBLE)
                views.setInt(id, "setBackgroundResource", bg)
                views.setTextViewText(id,
                    "$param ${String.format(Locale.US, "%.2f", value)}")
            }
        }

        private fun renderBottom(
            context: Context, views: RemoteViews, settings: SettingsStore,
        ) {
            views.setTextViewText(R.id.widgetBottomLine,
                if (settings.reminderEnabled) {
                    val text = nextReminderText(settings)
                    context.getString(R.string.widget_next_reminder, text)
                } else context.getString(R.string.widget_no_reminder))
        }

        private fun bgFor(range: TargetRange?, value: Float): Int {
            val s = range?.classify(value) ?: Status.UNKNOWN
            return when (s) {
                Status.IN_RANGE -> R.drawable.pill_in_range
                Status.NEAR     -> R.drawable.pill_near
                Status.OUT      -> R.drawable.pill_out
                Status.UNKNOWN  -> R.drawable.pill_unknown
            }
        }

        /** Next clock-wall slot at the user's reminder time, formatted
         *  short ("Sat 09:00"). */
        private fun nextReminderText(settings: SettingsStore): String {
            val now = System.currentTimeMillis()
            val c = Calendar.getInstance().apply {
                timeInMillis = now
                set(Calendar.HOUR_OF_DAY, settings.reminderHour)
                set(Calendar.MINUTE, settings.reminderMinute)
                set(Calendar.SECOND, 0); set(Calendar.MILLISECOND, 0)
            }
            if (c.timeInMillis <= now) c.add(Calendar.DAY_OF_YEAR, 1)
            return DateUtils.formatDateTime(null, c.timeInMillis,
                DateUtils.FORMAT_SHOW_WEEKDAY or DateUtils.FORMAT_SHOW_TIME
                    or DateUtils.FORMAT_ABBREV_WEEKDAY)
        }
    }
}
