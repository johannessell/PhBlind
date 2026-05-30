package com.example.poolwatertester.reminder

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.util.Log
import java.util.Calendar

/** Schedules / cancels the recurring "test your water" alarm via the
 *  platform AlarmManager. We deliberately use `setInexactRepeating` —
 *  daily reminders don't need second-level precision and the inexact API
 *  doesn't require the SCHEDULE_EXACT_ALARM permission. Re-armed on boot
 *  by [BootCompletedReceiver]. */
object ReminderScheduler {
    private const val TAG = "ReminderScheduler"
    const val REQUEST_CODE = 4711
    const val ACTION_FIRE = "com.example.poolwatertester.REMINDER_FIRE"

    fun enable(ctx: Context, intervalDays: Int, hour: Int, minute: Int) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(ctx, ReminderReceiver::class.java).apply {
            action = ACTION_FIRE
        }
        val pi = PendingIntent.getBroadcast(
            ctx, REQUEST_CODE, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val first = nextTrigger(hour, minute)
        val intervalMs = intervalDays.coerceAtLeast(1).toLong() *
            AlarmManager.INTERVAL_DAY
        am.setInexactRepeating(AlarmManager.RTC_WAKEUP, first, intervalMs, pi)
        Log.i(TAG, "scheduled: first=$first interval=$intervalMs")
    }

    fun disable(ctx: Context) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(ctx, ReminderReceiver::class.java).apply {
            action = ACTION_FIRE
        }
        val pi = PendingIntent.getBroadcast(
            ctx, REQUEST_CODE, intent,
            PendingIntent.FLAG_NO_CREATE or PendingIntent.FLAG_IMMUTABLE
        )
        if (pi != null) {
            am.cancel(pi); pi.cancel()
            Log.i(TAG, "cancelled")
        }
    }

    /** Next clock-wall slot at hour:minute; today if still in the future,
     *  otherwise tomorrow. */
    private fun nextTrigger(hour: Int, minute: Int): Long {
        val now = System.currentTimeMillis()
        val c = Calendar.getInstance().apply {
            timeInMillis = now
            set(Calendar.HOUR_OF_DAY, hour)
            set(Calendar.MINUTE, minute)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        if (c.timeInMillis <= now) c.add(Calendar.DAY_OF_YEAR, 1)
        return c.timeInMillis
    }
}
