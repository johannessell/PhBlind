plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.chaquo.python)
}

android {
    namespace = "com.example.poolwatertester"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.poolwatertester"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Chaquopy: limit native libs to real-phone ABI (+ emulator if needed).
        // Without this every supported ABI is bundled and the APK balloons.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    // Chaquopy: Python version + pip wheels to bundle into the APK.
    // Python 3.10 chosen because Chaquopy's PyPI mirror has a prebuilt
    // opencv-python wheel only for cp38 and cp310 (no cp311). OpenCV 4.5.1.48
    // is the latest opencv-python wheel published for cp310-android_24_arm64.
    chaquopy {
        defaultConfig {
            version = "3.10"
            pip {
                install("numpy")
                install("opencv-python==4.5.1.48")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}