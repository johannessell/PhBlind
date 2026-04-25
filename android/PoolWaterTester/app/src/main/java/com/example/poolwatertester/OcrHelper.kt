package com.example.poolwatertester

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class OcrBlock(
    val text: String,
    val x: Int,
    val y: Int,
    val w: Int,
    val h: Int,
    val confidence: Float,
) {
    fun toMap(): Map<String, Any> = mapOf(
        "text" to text,
        "x" to x,
        "y" to y,
        "w" to w,
        "h" to h,
        "score" to confidence,
    )
}

object OcrHelper {
    private val recognizer =
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    /**
     * Runs ML Kit text recognition on the given bitmap. Returns one
     * OcrBlock per Element (word-level) — finer granularity than Line,
     * matches OpenOCR's typical output and makes per-cell assignment
     * straightforward.
     */
    suspend fun recognize(bitmap: Bitmap): List<OcrBlock> =
        suspendCancellableCoroutine { cont ->
            val image = InputImage.fromBitmap(bitmap, 0)
            recognizer.process(image)
                .addOnSuccessListener { result ->
                    val out = ArrayList<OcrBlock>()
                    for (block in result.textBlocks) {
                        for (line in block.lines) {
                            for (element in line.elements) {
                                val box: Rect = element.boundingBox ?: continue
                                out += OcrBlock(
                                    text = element.text,
                                    x = box.left,
                                    y = box.top,
                                    w = box.width(),
                                    h = box.height(),
                                    confidence = element.confidence ?: 0f,
                                )
                            }
                        }
                    }
                    cont.resume(out)
                }
                .addOnFailureListener { e -> cont.resumeWithException(e) }
        }
}
