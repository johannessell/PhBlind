package com.example.poolwatertester.reminder

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.widget.PoolWaterWidget

/** AlarmManager schedules don't survive a reboot, so we re-enqueue the
 *  user's reminder on BOOT_COMPLETED (registered in the manifest). We
 *  also poke the home-screen widget so its "next reminder" line stays
 *  accurate after a reboot. */
class BootCompletedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        val store = SettingsStore.forActive(context)
        if (store.reminderEnabled) {
            ReminderScheduler.enable(
                context, store.reminderIntervalDays,
                store.reminderHour, store.reminderMinute
            )
        }
        PoolWaterWidget.refreshAll(context)
    }
}
