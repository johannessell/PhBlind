package com.example.poolwatertester.data

/** A named pool. The `id` is a UUID so renames don't touch the storage
 *  keys; `name` is what the user sees. */
data class Profile(val id: String, val name: String)
