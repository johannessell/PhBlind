package com.example.poolwatertester

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
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

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera()
        else requestPermission.launch(Manifest.permission.CAMERA)
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
                    cameraExecutor.execute {
                        val rotated = rotateBitmap(bitmap, rotation)
                        val rgba = bitmapToRgbaBytes(rotated)
                        try {
                            val res = measurement.callAttr(
                                "measure_rgba", rgba, rotated.width, rotated.height
                            )
                            val text = formatResults(res)
                            runOnUiThread {
                                binding.results.text = text
                                binding.measureButton.isEnabled = true
                                binding.measureButton.text = "Reset"
                                locked = true
                                measuring = false
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "measure failed", e)
                            runOnUiThread {
                                binding.results.text = "error: ${e.message}"
                                binding.measureButton.isEnabled = true
                                measuring = false
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
