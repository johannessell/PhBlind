package com.example.poolwatertester.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.example.poolwatertester.R
import com.example.poolwatertester.ReferenceActivity
import com.example.poolwatertester.data.HistoryEntry
import com.example.poolwatertester.data.HistoryStore
import com.example.poolwatertester.data.SettingsStore
import com.example.poolwatertester.data.Status
import com.example.poolwatertester.databinding.FragmentMeasureBinding
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Live camera preview + on-demand measurement. All the behaviour that used
 *  to live in MainActivity moved here verbatim; only the lifecycle hooks
 *  and `this`-as-Context references changed to fit a Fragment. The result
 *  card now renders one colour-coded status pill per measured parameter
 *  against the user-configured target range. */
class MeasureFragment : Fragment() {

    private var _binding: FragmentMeasureBinding? = null
    private val binding get() = _binding!!

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analyzer: PyObject
    private lateinit var measurement: PyObject
    private lateinit var historyStore: HistoryStore
    private lateinit var settingsStore: SettingsStore
    private var imageCapture: ImageCapture? = null

    @Volatile private var measuring = false
    @Volatile private var locked = false

    /** Result + ts captured at success and consumed by the Save button.
     *  Discard / reset clears them without writing to history. */
    private var pendingRes: PyObject? = null
    private var pendingTs: Long = 0L

    /** Lazily-initialised TTS engine. We init on first Save (rather than
     *  onCreate) because the user might never enable the toggle. */
    private var tts: TextToSpeech? = null
    private var ttsReady: Boolean = false

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else _binding?.status?.text = getString(R.string.status_camera_denied)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val py = Python.getInstance()
        analyzer = py.getModule("analyzer")
        measurement = py.getModule("measurement")
        historyStore = HistoryStore.forActive(requireContext())
        settingsStore = SettingsStore.forActive(requireContext())
        settingsStore.seedDefaultsIfMissing(SettingsStore.DEFAULT_RANGES.keys)
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMeasureBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.measureButton.setOnClickListener {
            if (!locked && !measuring) runMeasurement()
        }
        binding.saveButton.setOnClickListener {
            val res = pendingRes
            val ts = pendingTs
            if (res != null && ts > 0L) {
                appendToHistory(res, ts)
                snack(getString(R.string.snack_saved))
                if (settingsStore.ttsEnabled) speakResult(res)
                refreshRecentCard()
                com.example.poolwatertester.widget.PoolWaterWidget
                    .refreshAll(requireContext())
            }
            resetState()
        }
        binding.discardButton.setOnClickListener {
            snack(getString(R.string.snack_discarded))
            resetState()
        }
        binding.editReferenceButton.setOnClickListener {
            startActivity(Intent(requireContext(), ReferenceActivity::class.java))
        }
        binding.saveTrainingButton.setOnClickListener { saveTrainingFrame() }

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) requestPermission.launch(Manifest.permission.CAMERA)
    }

    private fun snack(msg: String) {
        val v = view ?: return
        com.google.android.material.snackbar.Snackbar
            .make(v, msg, com.google.android.material.snackbar.Snackbar.LENGTH_SHORT)
            .show()
    }

    /** Reads the per-parameter result aloud using the user's TTS engine.
     *  Lazily initialised on first call; ranges drive the spoken status. */
    private fun speakResult(res: PyObject) {
        if (tts == null) {
            tts = TextToSpeech(requireContext().applicationContext) { status ->
                ttsReady = status == TextToSpeech.SUCCESS
                if (ttsReady) doSpeak(res)
            }
        } else if (ttsReady) {
            doSpeak(res)
        }
    }

    private fun doSpeak(res: PyObject) {
        val resultsPy = res.asMap()[PyObject.fromJava("results")]?.asMap() ?: return
        val parts = StringBuilder()
        for ((k, v) in resultsPy) {
            val name = k.toString()
            val valuePy = v.asMap()[PyObject.fromJava("value")]
            val value = valuePy?.toString()?.toFloatOrNull() ?: continue
            val range = settingsStore.rangeFor(name)
            val statusWords = when (range?.classify(value)) {
                com.example.poolwatertester.data.Status.IN_RANGE -> "in range"
                com.example.poolwatertester.data.Status.NEAR -> "near limit"
                com.example.poolwatertester.data.Status.OUT -> "out of range"
                else -> ""
            }
            parts.append(name).append(' ')
                .append(String.format(java.util.Locale.US, "%.2f", value))
            if (statusWords.isNotEmpty()) parts.append(", ").append(statusWords)
            parts.append(". ")
        }
        tts?.speak(parts.toString(), TextToSpeech.QUEUE_FLUSH, null, "pwt")
    }

    private fun showSavedControls() {
        val b = _binding ?: return
        b.liveControls.visibility = View.GONE
        b.savedControls.visibility = View.VISIBLE
        b.measureButton.isEnabled = true   // doesn't matter; hidden
    }

    private fun showLiveControls() {
        val b = _binding ?: return
        b.savedControls.visibility = View.GONE
        b.liveControls.visibility = View.VISIBLE
        b.measureButton.text = getString(R.string.measure)
        b.measureButton.isEnabled = true
    }

    override fun onResume() {
        super.onResume()
        reloadReference()
        refreshRecentCard()
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera()
    }

    /** Surface the last saved measurement at the top so the user sees the
     *  baseline before lining up the next reading. Hidden when there's no
     *  history. */
    private fun refreshRecentCard() {
        val b = _binding ?: return
        val entries = historyStore.loadAll()
        val last = entries.lastOrNull()
        if (last == null) {
            b.recentCard.visibility = View.GONE
            return
        }
        b.recentCard.visibility = View.VISIBLE
        val when_ = android.text.format.DateUtils.getRelativeTimeSpanString(
            last.ts, System.currentTimeMillis(),
            android.text.format.DateUtils.MINUTE_IN_MILLIS
        )
        b.recentTitle.text = "${getString(R.string.recent_card_title)}  •  $when_"
        b.recentPills.removeAllViews()
        val ctx = requireContext()
        for ((param, value) in last.results) {
            val range = settingsStore.rangeFor(param)
            val status = range?.classify(value) ?: com.example.poolwatertester.data.Status.UNKNOWN
            b.recentPills.addView(buildPill(ctx, param, value, status))
        }
    }

    private fun buildPill(
        ctx: android.content.Context, param: String, value: Float,
        status: com.example.poolwatertester.data.Status,
    ): View {
        val (symbol, bg) = when (status) {
            com.example.poolwatertester.data.Status.IN_RANGE ->
                getString(R.string.pill_in_range) to R.drawable.pill_in_range
            com.example.poolwatertester.data.Status.NEAR ->
                getString(R.string.pill_near) to R.drawable.pill_near
            com.example.poolwatertester.data.Status.OUT ->
                getString(R.string.pill_out) to R.drawable.pill_out
            else ->
                getString(R.string.pill_unknown) to R.drawable.pill_unknown
        }
        val text = "$param ${String.format(java.util.Locale.US, "%.2f", value)}  $symbol"
        return TextView(ctx).apply {
            this.text = text
            setTextColor(android.graphics.Color.WHITE)
            textSize = 12f
            setBackgroundResource(bg)
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            lp.marginEnd = (resources.displayMetrics.density * 6).toInt()
            layoutParams = lp
        }
    }

    private fun reloadReference() {
        val refDir = java.io.File(requireContext().filesDir, "reference").apply { mkdirs() }
        try {
            val info = measurement.callAttr("init", refDir.absolutePath)
            Log.i(TAG, "measurement init: $info")
            val infoMap = info.asMap()
            val tplW = infoMap[PyObject.fromJava("width")]?.toInt() ?: 0
            val tplH = infoMap[PyObject.fromJava("height")]?.toInt() ?: 0
            if (tplW > 0 && tplH > 0) {
                val ar = maxOf(tplW, tplH).toFloat() / minOf(tplW, tplH).toFloat()
                _binding?.overlay?.setTargetAspect(ar)
            }
        } catch (e: Exception) {
            Log.e(TAG, "measurement init failed", e)
        }
    }

    private fun startCamera() {
        val ctx = context ?: return
        val providerFuture = ProcessCameraProvider.getInstance(ctx)
        providerFuture.addListener({
            val b = _binding ?: return@addListener
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(b.preview.surfaceProvider)
            }
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(1920, 1080),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }
            imageCapture = ImageCapture.Builder()
                .setResolutionSelector(resolutionSelector)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    viewLifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, analysis, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
                b.status.text = getString(R.string.status_bind_failed, e.message ?: "")
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    /** Pick the most useful live status string. Priority: searching →
     *  too far / too close / too tilted → tracking with progress. */
    private fun liveHint(
        found: Boolean, tilt: Float, areaFrac: Float, expectedArea: Float,
        progress: Int, required: Int,
    ): String {
        if (!found) return getString(R.string.status_searching)
        val low = 0.5f * expectedArea
        val high = 1.5f * expectedArea
        return when {
            areaFrac < low -> getString(R.string.status_card_too_far)
            areaFrac > high -> getString(R.string.status_card_too_close)
            tilt > 12f -> getString(R.string.status_card_too_tilted)
            else -> getString(R.string.status_tracking, progress, required)
        }
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (locked || measuring) { image.close(); return }
        try {
            val plane = image.planes[0]
            val buf = plane.buffer
            val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
            val res = measurement.callAttr(
                "find_quad_y",
                bytes, image.width, image.height,
                plane.rowStride, image.imageInfo.rotationDegrees
            )
            val m = res.asMap()
            val found = m[PyObject.fromJava("found")]?.toBoolean() ?: false
            val stable = m[PyObject.fromJava("stable")]?.toBoolean() ?: false
            val progress = m[PyObject.fromJava("progress")]?.toInt() ?: 0
            val required = m[PyObject.fromJava("required")]?.toInt() ?: 1
            val frameW = m[PyObject.fromJava("width")]?.toInt() ?: 0
            val frameH = m[PyObject.fromJava("height")]?.toInt() ?: 0
            val tilt = m[PyObject.fromJava("tilt_deg")]
                ?.toString()?.toFloatOrNull() ?: 0f
            val areaFrac = m[PyObject.fromJava("area_frac")]
                ?.toString()?.toFloatOrNull() ?: 0f
            val expectedArea = m[PyObject.fromJava("expected_area_frac")]
                ?.toString()?.toFloatOrNull() ?: 0.4f
            val quad = if (found) {
                val lst = m[PyObject.fromJava("quad")]!!.asList()
                FloatArray(8).also { arr ->
                    for (i in 0 until 4) {
                        val pt = lst[i].asList()
                        arr[i * 2] = pt[0].toFloat()
                        arr[i * 2 + 1] = pt[1].toFloat()
                    }
                }
            } else null

            val hint = liveHint(found, tilt, areaFrac, expectedArea,
                progress, required)
            postUi {
                val b = _binding ?: return@postUi
                b.overlay.update(quad, frameW, frameH, progress, required, stable)
                b.status.text = hint
            }

            if (stable && !measuring && !locked) {
                measuring = true
                postUi { runMeasurement() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
        } finally {
            image.close()
        }
    }

    private fun resetState() {
        locked = false
        measuring = false
        pendingRes = null
        pendingTs = 0L
        measurement.callAttr("reset_stability")
        val b = _binding ?: return
        b.overlay.clear()
        b.resultImage.visibility = View.GONE
        b.resultsCard.visibility = View.GONE
        b.results.visibility = View.GONE
        b.results.text = ""
        b.resultsRows.visibility = View.GONE
        b.resultsRows.removeAllViews()
        showLiveControls()
        b.status.text = getString(R.string.status_searching)
    }

    private fun runMeasurement() {
        val capture = imageCapture ?: return
        measuring = true
        val b = _binding ?: return
        b.measureButton.isEnabled = false
        showStatusText(getString(R.string.status_measuring))

        capture.takePicture(
            ContextCompat.getMainExecutor(requireContext()),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = image.toBitmap()
                    val rotation = image.imageInfo.rotationDegrees
                    image.close()
                    val rotated = rotateBitmap(bitmap, rotation)
                    postUi {
                        val bb = _binding ?: return@postUi
                        bb.overlay.clear()
                        bb.resultImage.setImageBitmap(rotated)
                        bb.resultImage.visibility = View.VISIBLE
                        locked = true
                    }
                    cameraExecutor.execute {
                        val rgba = bitmapToRgbaBytes(rotated)
                        try {
                            val res = measurement.callAttr(
                                "measure_rgba", rgba, rotated.width, rotated.height
                            )
                            val ts = logMeasurement(rotated, res)
                            if (isPlausible(res)) {
                                // Hold the result for the user's Save decision;
                                // do NOT auto-append to history.
                                pendingRes = res
                                pendingTs = ts
                                val overlay = decodeOverlay(res)
                                postUi {
                                    val bb = _binding ?: return@postUi
                                    if (overlay != null)
                                        bb.resultImage.setImageBitmap(overlay)
                                    showResultRows(res)
                                    showSavedControls()
                                    locked = true
                                    measuring = false
                                }
                            } else {
                                val msg = errorMessage(res)
                                    ?: getString(R.string.status_no_clear_reading)
                                postUi {
                                    val bb = _binding ?: return@postUi
                                    bb.resultImage.visibility = View.GONE
                                    bb.resultsCard.visibility = View.GONE
                                    bb.results.visibility = View.GONE
                                    bb.resultsRows.visibility = View.GONE
                                    showLiveControls()
                                    bb.status.text = msg
                                    locked = false
                                    measuring = false
                                    measurement.callAttr("reset_stability")
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "measure failed", e)
                            logMeasurement(rotated, null,
                                "${e.javaClass.simpleName}: ${e.message}")
                            postUi {
                                val bb = _binding ?: return@postUi
                                bb.resultImage.visibility = View.GONE
                                showStatusText(getString(R.string.status_error, e.message ?: ""))
                                showLiveControls()
                                locked = false
                                measuring = false
                                measurement.callAttr("reset_stability")
                            }
                        }
                    }
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "capture failed", exc)
                    val bb = _binding ?: return
                    showStatusText(getString(R.string.status_capture_error, exc.message ?: ""))
                    bb.measureButton.isEnabled = true
                    measuring = false
                }
            })
    }

    /** Plausible = card found, grid re-detected OK, at least one
     *  parameter measured AND no strict parameter (e.g. pH) reported an
     *  error. A strict-projection error means a known-biased fallback
     *  would have to be used — refuse to save and force a retake. */
    private fun isPlausible(res: PyObject): Boolean {
        val m = res.asMap()
        val found = m[PyObject.fromJava("found")]?.toBoolean() ?: false
        if (!found) return false
        val gridOk = m[PyObject.fromJava("grid_ok")]?.toBoolean() ?: false
        if (!gridOk) return false
        val errors = m[PyObject.fromJava("errors")]?.asMap()
        if (errors != null && errors.isNotEmpty()) return false
        val results = m[PyObject.fromJava("results")]?.asMap() ?: return false
        return results.isNotEmpty()
    }

    /** Returns a user-facing message describing which strict parameters
     *  failed, or null if the result has no error entries. */
    private fun errorMessage(res: PyObject): String? {
        val errors = res.asMap()[PyObject.fromJava("errors")]?.asMap()
            ?: return null
        if (errors.isEmpty()) return null
        val params = errors.keys.joinToString(", ") { it.toString() }
        return getString(R.string.status_param_unreliable, params)
    }

    private fun decodeOverlay(res: PyObject): Bitmap? {
        val jpg = res.asMap()[PyObject.fromJava("grid_overlay_jpg")] ?: return null
        return try {
            val bytes = jpg.toJava(ByteArray::class.java)
            if (bytes.isEmpty()) null
            else BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            Log.w(TAG, "overlay decode failed", e); null
        }
    }

    /** Status text (measuring / error / capture failure) goes in the
     *  single TextView; per-parameter rows are hidden in this mode. */
    private fun showStatusText(msg: String) {
        val b = _binding ?: return
        b.results.text = msg
        b.results.visibility = View.VISIBLE
        b.resultsRows.visibility = View.GONE
        b.resultsRows.removeAllViews()
        b.resultsCard.visibility = View.VISIBLE
    }

    /** Build one row per measured parameter: "name  value" + colour-coded
     *  status pill against the configured target range. Colour AND symbol
     *  are always paired for colour-blind accessibility. */
    private fun showResultRows(res: PyObject) {
        val b = _binding ?: return
        val results = res.asMap()[PyObject.fromJava("results")]?.asMap() ?: return
        b.resultsRows.removeAllViews()
        val ctx = requireContext()
        for ((k, v) in results) {
            val param = k.toString()
            val valuePy = v.asMap()[PyObject.fromJava("value")]
            val value = valuePy?.toString()?.toFloatOrNull() ?: continue
            val range = settingsStore.rangeFor(param)
            val status = range?.classify(value) ?: Status.UNKNOWN
            b.resultsRows.addView(buildResultRow(ctx, param, value, status))
        }
        b.resultsRows.visibility = View.VISIBLE
        b.results.visibility = View.GONE
        b.resultsCard.visibility = View.VISIBLE
    }

    private fun buildResultRow(ctx: android.content.Context,
                               param: String, value: Float, status: Status): View {
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = 4; bottomMargin = 4 }
        }
        val label = TextView(ctx).apply {
            text = String.format("%-5s  %.2f", param, value)
            setTextColor(Color.WHITE)
            textSize = 18f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
        }
        val (symbol, bg) = when (status) {
            Status.IN_RANGE -> "✓ OK"   to R.drawable.pill_in_range
            Status.NEAR     -> "⚠ NEAR" to R.drawable.pill_near
            Status.OUT      -> "✕ OUT"  to R.drawable.pill_out
            Status.UNKNOWN  -> "?"           to R.drawable.pill_unknown
        }
        val pill = TextView(ctx).apply {
            text = symbol
            setTextColor(Color.WHITE)
            textSize = 12f
            setBackgroundResource(bg)
            gravity = Gravity.CENTER
        }
        row.addView(label)
        row.addView(pill)
        return row
    }

    private fun logMeasurement(input: Bitmap, res: PyObject?, error: String? = null): Long {
        val ts = System.currentTimeMillis()
        try {
            val dir = java.io.File(
                requireContext().getExternalFilesDir(null), "measurements/$ts"
            ).apply { mkdirs() }
            java.io.FileOutputStream(java.io.File(dir, "input.jpg")).use {
                input.compress(Bitmap.CompressFormat.JPEG, 92, it)
            }
            if (res != null) {
                val map = res.asMap()
                val jpgPy = map[PyObject.fromJava("grid_overlay_jpg")]
                if (jpgPy != null) {
                    val bytes = jpgPy.toJava(ByteArray::class.java)
                    if (bytes.isNotEmpty()) {
                        java.io.FileOutputStream(
                            java.io.File(dir, "overlay.jpg")
                        ).use { it.write(bytes) }
                    }
                }
                val sb = StringBuilder()
                sb.append("ts=$ts\n")
                for (k in listOf("found", "grid_ok", "method",
                                 "grid_status", "quad", "results")) {
                    sb.append("$k=").append(
                        map[PyObject.fromJava(k)]?.toString() ?: "null"
                    ).append('\n')
                }
                java.io.FileOutputStream(java.io.File(dir, "result.txt")).use {
                    it.write(sb.toString().toByteArray())
                }
            }
            if (error != null) {
                java.io.FileOutputStream(java.io.File(dir, "error.txt")).use {
                    it.write(error.toByteArray())
                }
            }
            Log.i(TAG, "logged measurement: ${dir.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "log failed", e)
        }
        return ts
    }

    private fun appendToHistory(res: PyObject, ts: Long) {
        try {
            val resultsPy = res.asMap()[PyObject.fromJava("results")]?.asMap() ?: return
            val values = mutableMapOf<String, Float>()
            for ((k, v) in resultsPy) {
                val name = k.toString()
                val valuePy = v.asMap()[PyObject.fromJava("value")]
                val value = valuePy?.toString()?.toFloatOrNull() ?: continue
                values[name] = value
            }
            if (values.isNotEmpty()) {
                historyStore.append(
                    HistoryEntry(ts, values, "measurements/$ts")
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "history append failed", e)
        }
    }

    private fun saveTrainingFrame() {
        val capture = imageCapture ?: return
        val b = _binding ?: return
        b.saveTrainingButton.isEnabled = false
        val dir = java.io.File(requireContext().getExternalFilesDir(null), "training")
            .apply { mkdirs() }
        val name = "frame_${System.currentTimeMillis()}.jpg"
        val outFile = java.io.File(dir, name)
        val outOpts = ImageCapture.OutputFileOptions.Builder(outFile).build()
        capture.takePicture(
            outOpts,
            ContextCompat.getMainExecutor(requireContext()),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                    Log.i(TAG, "training frame saved: ${outFile.absolutePath}")
                    _binding?.let {
                        it.status.text = "saved ${outFile.name}"
                        it.saveTrainingButton.isEnabled = true
                    }
                }
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "training save failed", exc)
                    _binding?.let {
                        it.status.text = "save error: ${exc.message}"
                        it.saveTrainingButton.isEnabled = true
                    }
                }
            }
        )
    }

    private fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return src
        val m = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun bitmapToRgbaBytes(bmp: Bitmap): ByteArray {
        val buf = ByteBuffer.allocate(bmp.byteCount)
        bmp.copyPixelsToBuffer(buf)
        return buf.array()
    }

    /** Camera-thread → UI-thread helper. View may be torn down by the time
     *  the runnable runs (orientation change, tab switch); callers must
     *  re-check `_binding != null` inside. */
    private fun postUi(block: () -> Unit) {
        val act = activity ?: return
        act.runOnUiThread(block)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts?.shutdown(); tts = null
    }

    companion object {
        private const val TAG = "MeasureFragment"
    }
}
