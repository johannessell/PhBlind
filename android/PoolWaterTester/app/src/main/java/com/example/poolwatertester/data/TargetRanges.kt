package com.example.poolwatertester.data

/** A user-defined "good range" for one measured parameter. `ideal` is the
 *  optimum the user is shooting for; `min`..`max` is the acceptable band. */
data class TargetRange(
    val min: Float,
    val ideal: Float,
    val max: Float,
) {
    /** Width of the acceptable band; used to size the NEAR tolerance. */
    val band: Float get() = (max - min).coerceAtLeast(0.001f)

    fun classify(value: Float): Status {
        if (value.isNaN()) return Status.UNKNOWN
        if (value in min..max) return Status.IN_RANGE
        val tolerance = 0.25f * band
        return if (value in (min - tolerance)..(max + tolerance)) Status.NEAR
        else Status.OUT
    }
}

enum class Status { IN_RANGE, NEAR, OUT, UNKNOWN }
