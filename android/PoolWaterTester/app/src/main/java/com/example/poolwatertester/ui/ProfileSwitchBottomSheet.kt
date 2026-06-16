package com.example.poolwatertester.ui

import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.widget.NestedScrollView
import com.example.poolwatertester.R
import com.example.poolwatertester.data.Profile
import com.example.poolwatertester.data.ProfilesStore
import com.google.android.material.bottomsheet.BottomSheetDialogFragment

/** Lists every pool profile, lets the user switch / add / rename / delete.
 *  Selecting a different profile dismisses the sheet and invokes the
 *  caller-supplied [onSwitched] callback so the host can `recreate()`. */
class ProfileSwitchBottomSheet : BottomSheetDialogFragment() {

    private var onSwitched: (() -> Unit)? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        val ctx = requireContext()
        val scroll = NestedScrollView(ctx).apply {
            setPadding(dp(20), dp(20), dp(20), dp(20))
        }
        val root = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL }
        scroll.addView(root)
        rebuild(root)
        return scroll
    }

    private fun rebuild(root: LinearLayout) {
        root.removeAllViews()
        val ctx = requireContext()
        val store = ProfilesStore(ctx)
        val activeId = store.activeId()

        val title = TextView(ctx).apply {
            text = getString(R.string.profile_switcher_title)
            textSize = 18f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, 0, 0, dp(12))
        }
        root.addView(title)

        for (p in store.list()) {
            root.addView(rowFor(p, p.id == activeId, root))
        }

        val add = TextView(ctx).apply {
            text = "+  " + getString(R.string.profile_add)
            textSize = 16f
            setTextColor(Color.parseColor("#1976D2"))
            setPadding(0, dp(16), 0, dp(8))
            setOnClickListener {
                promptName(getString(R.string.profile_create_title), "") { name ->
                    val p = store.create(name)
                    store.setActive(p.id)
                    onSwitched?.invoke(); dismiss()
                }
            }
        }
        root.addView(add)
    }

    private fun rowFor(p: Profile, isActive: Boolean, parent: LinearLayout): View {
        val ctx = requireContext()
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(10), 0, dp(10))
            isClickable = true
            setOnClickListener {
                if (!isActive) {
                    ProfilesStore(ctx).setActive(p.id)
                    onSwitched?.invoke()
                }
                dismiss()
            }
        }
        val label = TextView(ctx).apply {
            text = if (isActive) "● ${p.name}" else "○ ${p.name}"
            textSize = 16f
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
        }
        val rename = TextView(ctx).apply {
            text = getString(R.string.profile_rename)
            setTextColor(Color.parseColor("#1976D2"))
            setPadding(dp(8), 0, dp(8), 0)
            setOnClickListener {
                promptName(getString(R.string.profile_rename_title), p.name) { name ->
                    ProfilesStore(ctx).rename(p.id, name)
                    rebuild(parent)
                }
            }
        }
        val delete = TextView(ctx).apply {
            text = getString(R.string.profile_delete)
            setTextColor(Color.parseColor("#C62828"))
            setPadding(dp(8), 0, 0, 0)
            setOnClickListener {
                AlertDialog.Builder(ctx)
                    .setTitle(R.string.confirm_delete_profile_title)
                    .setMessage(R.string.confirm_delete_profile_message)
                    .setNegativeButton(R.string.cancel, null)
                    .setPositiveButton(R.string.profile_delete) { _, _ ->
                        val wasActive = (p.id == ProfilesStore(ctx).activeId())
                        ProfilesStore(ctx).delete(p.id)
                        if (wasActive) onSwitched?.invoke()
                        rebuild(parent)
                    }
                    .show()
            }
        }
        row.addView(label)
        row.addView(rename)
        row.addView(delete)
        return row
    }

    private fun promptName(title: String, initial: String, onOk: (String) -> Unit) {
        val ctx = requireContext()
        val edit = EditText(ctx).apply {
            setText(initial)
            hint = getString(R.string.profile_name_hint)
            setSelection(initial.length)
        }
        val container = LinearLayout(ctx).apply {
            setPadding(dp(20), dp(8), dp(20), 0)
            addView(edit)
        }
        AlertDialog.Builder(ctx)
            .setTitle(title)
            .setView(container)
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(R.string.ok) { _, _ ->
                val name = edit.text.toString().trim().ifEmpty { return@setPositiveButton }
                onOk(name)
            }
            .show()
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    companion object {
        fun newInstance(onSwitched: () -> Unit): ProfileSwitchBottomSheet =
            ProfileSwitchBottomSheet().also { it.onSwitched = onSwitched }
    }
}
