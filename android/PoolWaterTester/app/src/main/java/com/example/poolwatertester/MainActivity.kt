package com.example.poolwatertester

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
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
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.poolwatertester.databinding.ActivityMainBinding
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analyzer: PyObject
    private lateinit var measurement: PyObject
    private var imageCapture: ImageCapture? = null
    @Volatile private var measuring = false
    @Volatile private var locked = false

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else binding.status.text = "camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.main) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        val py = Python.getInstance()
        analyzer = py.getModule("analyzer")
        measurement = py.getModule("measurement")
        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.measureButton.setOnClickListener {
            if (locked) resetState() else runMeasurement()
        }
        binding.editReferenceButton.setOnClickListener {
            startActivity(android.content.Intent(this, ReferenceActivity::class.java))
        }
        binding.saveTrainingButton.setOnClickListener { saveTrainingFrame() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) requestPermission.launch(Manifest.permission.CAMERA)
        // Camera bind happens in onResume — Android lifecycle on return from
        // ReferenceActivity is A.onResume → B.onStop, so B still holds the
        // camera here. Re-binding in onResume (which calls unbindAll first)
        // is the only reliable way to re-acquire after navigation.
    }

    override fun onResume() {
        super.onResume()
        reloadReference()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera()
    }

    private fun reloadReference() {
        // Re-runs each onResume so a reference saved in ReferenceActivity is
        // picked up when the user returns. Also pushes the template's aspect
        // ratio into the overlay's guide rectangle.
        val refDir = java.io.File(filesDir, "reference").apply { mkdirs() }
        try {
            val info = measurement.callAttr("init", refDir.absolutePath)
            Log.i(TAG, "measurement init: $info")
            val infoMap = info.asMap()
            val tplW = infoMap[PyObject.fromJava("width")]?.toInt() ?: 0
            val tplH = infoMap[PyObject.fromJava("height")]?.toInt() ?: 0
            if (tplW > 0 && tplH > 0) {
                val ar = maxOf(tplW, tplH).toFloat() / minOf(tplW, tplH).toFloat()
                binding.overlay.setTargetAspect(ar)
            }
        } catch (e: Exception) {
            Log.e(TAG, "measurement init failed", e)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.preview.surfaceProvider)
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
                    this, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, analysis, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
                binding.status.text = "bind failed: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
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

            runOnUiThread {
                binding.overlay.update(quad, frameW, frameH, progress, required, stable)
                binding.status.text = if (found)
                    "tracking ${progress}/${required}"
                else "searching..."
            }

            if (stable && !measuring && !locked) {
                measuring = true
                runOnUiThread { runMeasurement() }
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
        measurement.callAttr("reset_stability")
        binding.overlay.clear()
        binding.resultImage.visibility = View.GONE
        binding.results.visibility = View.GONE
        binding.results.text = ""
        binding.measureButton.text = "Measure"
        binding.status.text = "searching..."
    }

    private fun runMeasurement() {
        val capture = imageCapture ?: return
        measuring = true
        binding.measureButton.isEnabled = false
        binding.results.text = "measuring..."
        binding.results.visibility = View.VISIBLE

        capture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = image.toBitmap()
                    val rotation = image.imageInfo.rotationDegrees
                    image.close()
                    val rotated = rotateBitmap(bitmap, rotation)
                    // Freeze the view on the captured frame: stop showing the
                    // live preview the moment the picture is taken.
                    runOnUiThread {
                        binding.overlay.clear()
                        binding.resultImage.setImageBitmap(rotated)
                        binding.resultImage.visibility = View.VISIBLE
                        locked = true   // halt the analyzer
                    }
                    cameraExecutor.execute {
                        val rgba = bitmapToRgbaBytes(rotated)
                        try {
                            val res = measurement.callAttr(
                                "measure_rgba", rgba, rotated.width, rotated.height
                            )
                            if (isPlausible(res)) {
                                val overlay = decodeOverlay(res)
                                val text = formatResults(res)
                                runOnUiThread {
                                    if (overlay != null)
                                        binding.resultImage.setImageBitmap(overlay)
                                    binding.results.text = text
                                    binding.results.visibility = View.VISIBLE
                                    binding.measureButton.isEnabled = true
                                    binding.measureButton.text = "Reset"
                                    locked = true
                                    measuring = false
                                }
                            } else {
                                // No plausible reading — drop the frozen frame
                                // and return to the live view to try again.
                                runOnUiThread {
                                    binding.resultImage.visibility = View.GONE
                                    binding.results.visibility = View.GONE
                                    binding.measureButton.isEnabled = true
                                    binding.measureButton.text = "Measure"
                                    binding.status.text = "no clear reading — keep steady"
                                    locked = false
                                    measuring = false
                                    measurement.callAttr("reset_stability")
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "measure failed", e)
                            runOnUiThread {
                                binding.resultImage.visibility = View.GONE
                                binding.results.text = "error: ${e.message}"
                                binding.results.visibility = View.VISIBLE
                                binding.measureButton.isEnabled = true
                                locked = false
                                measuring = false
                                measurement.callAttr("reset_stability")
                            }
                        }
                    }
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "capture failed", exc)
                    binding.results.text = "capture error: ${exc.message}"
                    binding.measureButton.isEnabled = true
                    measuring = false
                }
            })
    }

    /** Plausible = card found, grid re-detected OK, at least one parameter
     *  measured. Drives "show result" vs "return to live view". */
    private fun isPlausible(res: PyObject): Boolean {
        val m = res.asMap()
        val found = m[PyObject.fromJava("found")]?.toBoolean() ?: false
        if (!found) return false
        val gridOk = m[PyObject.fromJava("grid_ok")]?.toBoolean() ?: false
        if (!gridOk) return false
        val results = m[PyObject.fromJava("results")]?.asMap() ?: return false
        return results.isNotEmpty()
    }

    /** Decode the warped+grid overlay JPEG returned by measure_rgba. */
    private fun decodeOverlay(res: PyObject): Bitmap? {
        val jpg = res.asMap()[PyObject.fromJava("grid_overlay_jpg")]
            ?: return null
        return try {
            val bytes = jpg.toJava(ByteArray::class.java)
            if (bytes.isEmpty()) null
            else BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            Log.w(TAG, "overlay decode failed", e)
            null
        }
    }

    private fun saveTrainingFrame() {
        // Captures the full-res sensor frame and writes it to the app's
        // external files dir as JPG. Works regardless of detection state, so
        // the user can specifically save frames where the live tracker
        // misbehaves. Pull with:
        //   adb pull /sdcard/Android/data/com.example.poolwatertester/files/training/
        val capture = imageCapture ?: return
        binding.saveTrainingButton.isEnabled = false
        val dir = java.io.File(getExternalFilesDir(null), "training").apply { mkdirs() }
        val name = "frame_${System.currentTimeMillis()}.jpg"
        val outFile = java.io.File(dir, name)
        val outOpts = ImageCapture.OutputFileOptions.Builder(outFile).build()
        capture.takePicture(
            outOpts,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                    Log.i(TAG, "training frame saved: ${outFile.absolutePath}")
                    binding.status.text = "saved ${outFile.name}"
                    binding.saveTrainingButton.isEnabled = true
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "training save failed", exc)
                    binding.status.text = "save error: ${exc.message}"
                    binding.saveTrainingButton.isEnabled = true
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

    private fun formatResults(res: PyObject): String {
        val map = res.asMap()
        val found = map[PyObject.fromJava("found")]?.toBoolean() ?: false
        if (!found) return "no indicator detected"
        val results = map[PyObject.fromJava("results")]?.asMap() ?: return "no results"
        if (results.isEmpty()) return "detected, but no params measured"
        val sb = StringBuilder()
        for ((k, v) in results) {
            val vm = v.asMap()
            val value = vm[PyObject.fromJava("value")]
            val ch = vm[PyObject.fromJava("channel")]
            val r = vm[PyObject.fromJava("r")]
            sb.append(String.format("%-6s %s   [ch=%s r=%s]\n",
                k.toString() + ":", value.toString(), ch.toString(), r.toString()))
        }
        return sb.toString().trimEnd()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "PyBoot"
    }
}
