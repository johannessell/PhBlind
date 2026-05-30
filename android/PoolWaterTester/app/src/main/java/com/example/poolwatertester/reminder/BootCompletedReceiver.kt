package com.example.poolwatertester.reminder

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.example.poolwatertester.data.SettingsStore

/** AlarmManager schedules don't survive a reboot, so we re-enqueue the
 *  user's reminder on BOOT_COMPLETED (registered in the manifest). */
class BootCompletedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        val store = SettingsStore(context)
        if (!store.reminderEnabled) return
        ReminderScheduler.enable(
            context, store.reminderIntervalDays,
            store.reminderHour, store.reminderMinute
        )
    }
}
