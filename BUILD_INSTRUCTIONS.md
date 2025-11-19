# Build Instructions

## Prerequisites

### Required Software
1. **Android Studio**: Arctic Fox (2020.3.1) or later
   - Download from: https://developer.android.com/studio

2. **Java Development Kit (JDK)**: Version 17
   - Included with Android Studio
   - Or download from: https://adoptium.net/

3. **Android SDK**: API Level 34
   - Installed via Android Studio SDK Manager

### System Requirements
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.14 or later
- **Linux**: Ubuntu 18.04 or later (64-bit)
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: 8 GB minimum

## Setup Steps

### 1. Clone or Download Project
```bash
git clone <repository-url>
cd FoodRecognizerApp
```

Or download and extract the ZIP file.

### 2. Open in Android Studio

1. Launch Android Studio
2. Click "Open" or "Open an existing Android Studio project"
3. Navigate to the FoodRecognizerApp directory
4. Click "OK"

### 3. Gradle Sync

Android Studio will automatically:
- Download required dependencies
- Sync Gradle files
- Index the project

Wait for this process to complete (shown in bottom status bar).

If sync fails:
- Click "File" > "Sync Project with Gradle Files"
- Or click the "Sync" button in the toolbar

### 4. Install SDK Components

If prompted, install:
- Android SDK Platform 34
- Android SDK Build-Tools
- Android Emulator (optional, for testing)

Access via: Tools > SDK Manager

## Building the App

### Method 1: Using Android Studio GUI

#### Debug Build
1. Click "Build" menu
2. Select "Build Bundle(s) / APK(s)"
3. Select "Build APK(s)"
4. Wait for build to complete
5. Click "locate" in the notification to find APK

Output: `app/build/outputs/apk/debug/app-debug.apk`

#### Release Build
1. Click "Build" menu
2. Select "Build Bundle(s) / APK(s)"
3. Select "Build APK(s)"
4. Select "release" build variant in Build Variants panel
5. Wait for build to complete

Output: `app/build/outputs/apk/release/app-release-unsigned.apk`

### Method 2: Using Command Line

#### On macOS/Linux

Debug build:
```bash
./gradlew assembleDebug
```

Release build:
```bash
./gradlew assembleRelease
```

Clean build:
```bash
./gradlew clean assembleDebug
```

#### On Windows

Debug build:
```cmd
gradlew.bat assembleDebug
```

Release build:
```cmd
gradlew.bat assembleRelease
```

### Method 3: Using GitHub Actions

1. Push code to GitHub repository
2. Go to "Actions" tab
3. Workflow runs automatically on push
4. Download APK from "Artifacts" section

## Running the App

### On Physical Device

1. Enable Developer Options on your Android device:
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times

2. Enable USB Debugging:
   - Go to Settings > Developer Options
   - Enable "USB Debugging"

3. Connect device via USB

4. In Android Studio:
   - Select your device from device dropdown
   - Click "Run" (green play button)

### On Emulator

1. Create emulator:
   - Tools > Device Manager
   - Click "Create Device"
   - Select a device (e.g., Pixel 5)
   - Select system image (API 34)
   - Finish setup

2. Start emulator and run app:
   - Select emulator from device dropdown
   - Click "Run"

### Install APK Directly

1. Build APK using methods above
2. Transfer APK to device
3. Enable "Install from Unknown Sources"
4. Open APK file on device
5. Tap "Install"

## Signing Release APK

For production distribution, sign the APK:

### Generate Keystore
```bash
keytool -genkey -v -keystore my-release-key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias my-alias
```

### Configure Signing in build.gradle

Add to `app/build.gradle`:
```gradle
android {
    signingConfigs {
        release {
            storeFile file("my-release-key.jks")
            storePassword "your-password"
            keyAlias "my-alias"
            keyPassword "your-password"
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

### Build Signed APK
```bash
./gradlew assembleRelease
```

## Troubleshooting

### Gradle Build Failed

**Error**: "Could not find com.android.tools.build:gradle:X.X.X"
- Solution: Update Gradle version in gradle/wrapper/gradle-wrapper.properties
- Or: File > Project Structure > Project > Gradle Version

**Error**: "SDK location not found"
- Solution: Create local.properties file with:
  ```
  sdk.dir=/path/to/Android/sdk
  ```

### Dependency Resolution Failed

**Error**: "Could not resolve all dependencies"
- Solution: Check internet connection
- Or: File > Invalidate Caches / Restart
- Or: Delete .gradle folder and sync again

### Build Takes Too Long

- Enable offline mode: File > Settings > Build, Execution, Deployment > Gradle > Offline work
- Increase Gradle heap size in gradle.properties:
  ```
  org.gradle.jvmargs=-Xmx4096m
  ```

### OutOfMemoryError

Add to gradle.properties:
```
org.gradle.jvmargs=-Xmx4096m -XX:MaxPermSize=1024m -XX:+HeapDumpOnOutOfMemoryError
```

## Project Structure

```
FoodRecognizerApp/
├── app/
│   ├── build.gradle              # App module Gradle config
│   ├── proguard-rules.pro        # ProGuard rules
│   └── src/
│       └── main/
│           ├── AndroidManifest.xml
│           ├── java/             # Java source files
│           └── res/              # Resources (layouts, drawables, etc.)
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar    # Gradle wrapper JAR
│       └── gradle-wrapper.properties
├── build.gradle                  # Project Gradle config
├── settings.gradle               # Project settings
├── gradle.properties             # Gradle properties
├── gradlew                       # Gradle wrapper script (Unix)
└── gradlew.bat                   # Gradle wrapper script (Windows)
```

## Build Variants

### Debug
- Debuggable: Yes
- Optimized: No
- Signing: Debug keystore
- Purpose: Development and testing

### Release
- Debuggable: No
- Optimized: Yes (with R8/ProGuard)
- Signing: Release keystore (configure manually)
- Purpose: Production distribution

Switch variants: Build > Select Build Variant

## Continuous Integration

The project includes GitHub Actions workflow:
- Automatic builds on push to main/master/develop
- Runs tests
- Generates APK artifacts
- 30-day artifact retention

See: `.github/workflows/android.yml`

## Performance Optimization

### Faster Builds
1. Enable Gradle daemon (enabled by default)
2. Increase heap size in gradle.properties
3. Enable parallel builds:
   ```
   org.gradle.parallel=true
   org.gradle.configureondemand=true
   ```

### Reduce APK Size
1. Enable ProGuard/R8:
   ```gradle
   buildTypes {
       release {
           minifyEnabled true
           shrinkResources true
       }
   }
   ```

2. Use APK Analyzer: Build > Analyze APK

## Next Steps

After successful build:
1. Review USAGE.md for app usage instructions
2. Configure your API server
3. Test on physical device
4. Consider signing for production
5. Publish to Google Play Store (optional)

## Support

For build issues:
1. Check this document
2. Clean and rebuild project
3. Invalidate caches and restart
4. Check Android Studio logs
5. Search error messages online
6. Check Stack Overflow
