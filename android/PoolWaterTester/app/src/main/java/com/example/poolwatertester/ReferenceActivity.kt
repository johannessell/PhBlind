package com.example.poolwatertester

import android.Manifest
import android.content.pm.PackageManager
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
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
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.poolwatertester.databinding.ActivityReferenceBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

class ReferenceActivity : AppCompatActivity() {
    private lateinit var binding: ActivityReferenceBinding
    private lateinit var refBuilder: PyObject
    private lateinit var measurement: PyObject
    private var imageCapture: ImageCapture? = null

    private var refDict: PyObject? = null
    private var warpedBitmap: Bitmap? = null
    private var debugDir: File? = null
    private var debugIdx: Int = -1  // -1 = warped (default)

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else binding.refStatus.text = "camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityReferenceBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.refRoot) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        val py = Python.getInstance()
        refBuilder = py.getModule("reference_builder")
        measurement = py.getModule("measurement")

        binding.refCaptureButton.setOnClickListener { capture() }
        binding.refRetakeButton.setOnClickListener { showCaptureState() }
        binding.refSaveButton.setOnClickListener { save() }
        binding.refDebugButton.setOnClickListener { cycleDebugImage() }

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
                it.setSurfaceProvider(binding.refPreview.surfaceProvider)
            }
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(1920, 1080),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()
            imageCapture = ImageCapture.Builder()
                .setResolutionSelector(resolutionSelector)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
                binding.refStatus.text = "bind failed: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun capture() {
        val capture = imageCapture ?: return
        binding.refCaptureButton.isEnabled = false
        binding.refStatus.text = "capturing..."
        binding.refProgress.visibility = View.VISIBLE

        capture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bmp = image.toBitmap()
                    val rot = image.imageInfo.rotationDegrees
                    image.close()
                    lifecycleScope.launch { processCapture(bmp, rot) }
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "capture error", exc)
                    binding.refStatus.text = "capture error: ${exc.message}"
                    binding.refCaptureButton.isEnabled = true
                    binding.refProgress.visibility = View.GONE
                }
            })
    }

    private suspend fun processCapture(bmp: Bitmap, rotation: Int) {
        val dbg = File(filesDir, "reference_debug").apply {
            mkdirs()
            listFiles()?.forEach { it.delete() }
        }
        debugDir = dbg
        debugIdx = -1
        refBuilder.callAttr("set_debug_dir", dbg.absolutePath)
        Log.i(TAG, "debug images -> ${dbg.absolutePath}")
        try {
            val (ref, warped) = withContext(Dispatchers.Default) {
                val rotated = rotateBitmap(bmp, rotation)
                val rgba = bitmapToRgbaBytes(rotated)
                val refPy = refBuilder.callAttr(
                    "build_reference_from_rgba", rgba, rotated.width, rotated.height
                )
                val m = refPy.asMap()
                val jpgPy = m[PyObject.fromJava("warped_jpg")]
                val jpgBytes = jpgPy?.toJava(ByteArray::class.java) ?: ByteArray(0)
                val warpedBmp = BitmapFactory.decodeByteArray(jpgBytes, 0, jpgBytes.size)
                Pair(refPy, warpedBmp)
            }

            binding.refStatus.text = "reading labels..."
            val ocrBlocks = OcrHelper.recognize(warped)
            val blocksList = ocrBlocks.map { it.toMap() }

            val refWithOcr = withContext(Dispatchers.Default) {
                refBuilder.callAttr("assign_ocr_to_cells", ref, blocksList)
            }

            refDict = refWithOcr
            warpedBitmap = warped
            withContext(Dispatchers.IO) { exportDebugToDownloads(dbg) }
            showEditState(refWithOcr, warped)
        } catch (e: Exception) {
            Log.e(TAG, "processCapture failed", e)
            binding.refStatus.text = "error: ${e.message}"
            binding.editHint.text = "detection failed: ${e.message}"
            binding.refCaptureButton.isEnabled = true
            withContext(Dispatchers.IO) { exportDebugToDownloads(dbg) }
            showDebugOnly()
        } finally {
            binding.refProgress.visibility = View.GONE
        }
    }

    private fun exportDebugToDownloads(srcDir: File) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        val files = srcDir.listFiles()?.filter { it.name.endsWith(".jpg") } ?: return
        if (files.isEmpty()) return
        val resolver = contentResolver
        val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
        val relPath = "Download/PoolWaterTester"
        for (f in files) {
            try {
                resolver.query(
                    collection,
                    arrayOf(MediaStore.Downloads._ID),
                    "${MediaStore.Downloads.DISPLAY_NAME}=? AND ${MediaStore.Downloads.RELATIVE_PATH}=?",
                    arrayOf(f.name, "$relPath/"),
                    null
                )?.use { c ->
                    while (c.moveToNext()) {
                        val id = c.getLong(0)
                        val uri = android.content.ContentUris.withAppendedId(collection, id)
                        resolver.delete(uri, null, null)
                    }
                }
                val cv = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, f.name)
                    put(MediaStore.Downloads.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Downloads.RELATIVE_PATH, relPath)
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val uri = resolver.insert(collection, cv) ?: continue
                resolver.openOutputStream(uri)?.use { os ->
                    f.inputStream().use { it.copyTo(os) }
                }
                val done = ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }
                resolver.update(uri, done, null, null)
            } catch (e: Exception) {
                Log.w(TAG, "export ${f.name} failed", e)
            }
        }
        Log.i(TAG, "exported ${files.size} debug files to Download/PoolWaterTester/")
    }

    private fun cycleDebugImage() {
        val dir = debugDir ?: return
        val files = dir.listFiles()?.filter { it.name.endsWith(".jpg") }
            ?.sortedBy { it.name } ?: return
        if (files.isEmpty()) {
            binding.editHint.text = "no debug files"
            return
        }
        debugIdx += 1
        if (debugIdx >= files.size) {
            // Cycle back to warped
            debugIdx = -1
            warpedBitmap?.let { binding.refImage.setImageBitmap(it) }
            binding.cellEditor.visibility = View.VISIBLE
            binding.editHint.text = "review values, then Save"
            return
        }
        val f = files[debugIdx]
        val bmp = BitmapFactory.decodeFile(f.absolutePath)
        if (bmp != null) {
            binding.refImage.setImageBitmap(bmp)
            binding.cellEditor.visibility = View.GONE
            binding.editHint.text = f.name
        }
    }

    private fun showDebugOnly() {
        // Detection failed — drop into edit-state UI showing only debug images
        binding.captureState.visibility = View.GONE
        binding.editState.visibility = View.VISIBLE
        binding.cellEditor.visibility = View.GONE
        binding.refSaveButton.isEnabled = false
        debugIdx = -1
        cycleDebugImage()
    }

    private fun showCaptureState() {
        binding.captureState.visibility = View.VISIBLE
        binding.editState.visibility = View.GONE
        binding.refCaptureButton.isEnabled = true
        binding.refStatus.text = "hold reference card steady"
    }

    private fun showEditState(ref: PyObject, warped: Bitmap) {
        binding.captureState.visibility = View.GONE
        binding.editState.visibility = View.VISIBLE
        binding.cellEditor.visibility = View.VISIBLE
        binding.refSaveButton.isEnabled = true
        binding.editHint.text = "review values, then Save"
        debugIdx = -1
        binding.refImage.setImageBitmap(warped)

        val m = ref.asMap()
        val width = m[PyObject.fromJava("width")]!!.toInt()
        val height = m[PyObject.fromJava("height")]!!.toInt()
        val cellsPy = m[PyObject.fromJava("cells")]!!.asList()
        val paramsPy = m[PyObject.fromJava("parameters")]!!.asList()

        val cells = ArrayList<CellOverlayEditor.Cell>()
        val initialValues = HashMap<Int, String?>()
        for (cPy in cellsPy) {
            val cm = cPy.asMap()
            val cellIdx = cm[PyObject.fromJava("cell_idx")]!!.toInt()
            val rowIdx = cm[PyObject.fromJava("row_idx")]!!.toInt()
            val groupIdxPy = cm[PyObject.fromJava("group_idx")]
            val groupIdx = if (groupIdxPy == null || groupIdxPy.toString() == "None") null
                           else groupIdxPy.toInt()
            val isColor = cm[PyObject.fromJava("is_color_cell")]!!.toBoolean()
            val x = cm[PyObject.fromJava("x")]!!.toInt()
            val y = cm[PyObject.fromJava("y")]!!.toInt()
            val w = cm[PyObject.fromJava("w")]!!.toInt()
            val h = cm[PyObject.fromJava("h")]!!.toInt()
            cells += CellOverlayEditor.Cell(cellIdx, rowIdx, groupIdx, isColor, x, y, w, h)
            val vPy = cm[PyObject.fromJava("value")]
            initialValues[cellIdx] = if (vPy == null || vPy.toString() == "None") null
                                     else vPy.toString()
        }

        val headers = ArrayList<CellOverlayEditor.ParamHeader>()
        val initialNames = HashMap<Int, String?>()
        for (pPy in paramsPy) {
            val pm = pPy.asMap()
            val gi = pm[PyObject.fromJava("group_idx")]!!.toInt()
            val namePy = pm[PyObject.fromJava("name")]
            initialNames[gi] = if (namePy == null || namePy.toString() == "None") null
                               else namePy.toString()
            val groupCells = cells.filter { it.groupIdx == gi }
            if (groupCells.isEmpty()) continue
            val xMin = groupCells.minOf { it.x }
            val xMax = groupCells.maxOf { it.x + it.w }
            val yTop = groupCells.minOf { it.y }
            headers += CellOverlayEditor.ParamHeader(
                groupIdx = gi,
                xCenter = (xMin + xMax) / 2,
                yTop = yTop,
                width = xMax - xMin,
            )
        }

        binding.cellEditor.setReference(
            bitmapWidth = width,
            bitmapHeight = height,
            cells = cells,
            headers = headers,
            initialValues = initialValues,
            initialNames = initialNames,
        )
    }

    private fun save() {
        val ref = refDict ?: return
        val warped = warpedBitmap ?: return
        binding.refSaveButton.isEnabled = false
        binding.refProgress.visibility = View.VISIBLE

        lifecycleScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    val editedValues = binding.cellEditor.cellValues
                    val editedNames = binding.cellEditor.paramNames
                    refBuilder.callAttr("apply_edits_and_finalize", ref, editedValues, editedNames)

                    val rgba = bitmapToRgbaBytes(warped)
                    refBuilder.callAttr(
                        "compute_best_channels_rgba",
                        ref, rgba, warped.width, warped.height
                    )

                    val refDir = File(filesDir, "reference").apply { mkdirs() }
                    val jsonStr = refBuilder.callAttr("ref_to_json_str", ref).toString()
                    File(refDir, "reference.json").writeText(jsonStr)

                    val tplFile = File(refDir, "template.jpg")
                    FileOutputStream(tplFile).use { out ->
                        warped.compress(Bitmap.CompressFormat.JPEG, 95, out)
                    }

                    measurement.callAttr("init", refDir.absolutePath)
                }
                binding.refStatus.text = "saved"
                finish()
            } catch (e: Exception) {
                Log.e(TAG, "save failed", e)
                binding.refStatus.text = "save error: ${e.message}"
                binding.refSaveButton.isEnabled = true
            } finally {
                binding.refProgress.visibility = View.GONE
            }
        }
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

    companion object {
        private const val TAG = "RefEditor"
    }
}
