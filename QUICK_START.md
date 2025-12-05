# Quick Start Guide

## What You Got

A complete, production-ready Android app for Food-101 recognition with:
- Native Android implementation (Java)
- Material Design UI
- Camera & Gallery support
- API integration with OkHttp
- Top 5 predictions display
- GitHub Actions CI/CD

## Files Created: 30+

### Core Application
- 4 Java source files (MainActivity, ApiClient, Result, ResultAdapter)
- 2 Layout XML files (main screen, result cards)
- 8 Resource XML files (colors, strings, themes, etc.)
- AndroidManifest with all permissions

### Build System
- Complete Gradle configuration
- Gradle wrapper (gradlew, gradlew.bat)
- GitHub Actions workflow for automatic APK builds

### Documentation
- README.md - Main project documentation
- USAGE.md - User guide
- BUILD_INSTRUCTIONS.md - Build guide
- PROJECT_SUMMARY.md - Complete overview
- FILE_STRUCTURE.txt - Visual structure

## 3-Step Setup

### 1. Open in Android Studio
```bash
# Open Android Studio
# File > Open > Select FoodRecognizerApp folder
# Wait for Gradle sync
```

### 2. Build APK
```bash
# In terminal:
cd FoodRecognizerApp
./gradlew assembleDebug

# Or in Android Studio:
# Build > Build Bundle(s) / APK(s) > Build APK(s)
```

### 3. Run on Device
```bash
# Connect Android device via USB
# Enable USB Debugging
# Click Run button (green play icon)
```

## Using the App

1. Enter server URL: `http://192.168.1.100:5000`
2. Tap Camera or Gallery to select food image
3. Tap "Recognize Food" button
4. View top 5 predictions with confidence scores

## API Server Requirements

Your server must:
- Run on port 5000 (or configure as needed)
- Accept POST at `/predict` endpoint
- Handle multipart/form-data with field `image`
- Return JSON:
  ```json
  {
    "success": true,
    "predictions": [
      {"dish": "Pizza", "confidence": "85.5"},
      ...
    ]
  }
  ```

## GitHub Actions

Automatic APK builds:
1. Push code to GitHub
2. Go to Actions tab
3. Download APK artifacts (30-day retention)

## Key Features

- Modern Material Design UI
- Real-time food recognition
- Camera integration
- Gallery selection
- Server URL persistence
- Top 5 predictions display
- Error handling
- Permission management

## Technical Stack

- Min SDK: 24 (Android 7.0)
- Target SDK: 34 (Android 14)
- OkHttp 4.12.0 for API calls
- Gson 2.10.1 for JSON parsing
- Material Components 1.10.0
- RecyclerView for results list

## File Locations

```
Key files:
/app/src/main/java/com/example/foodrecognizer/MainActivity.java
/app/src/main/res/layout/activity_main.xml
/app/build.gradle
/build.gradle
/.github/workflows/android.yml
```

## Build Commands

```bash
# Debug build
./gradlew assembleDebug

# Release build
./gradlew assembleRelease

# Clean build
./gradlew clean assembleDebug

# Run tests
./gradlew test
```

## Output Locations

- Debug APK: `app/build/outputs/apk/debug/app-debug.apk`
- Release APK: `app/build/outputs/apk/release/app-release-unsigned.apk`

## Troubleshooting

### Build fails
```bash
./gradlew clean
# File > Invalidate Caches / Restart in Android Studio
```

### Permissions denied
- Check device settings
- Enable Camera/Storage permissions manually

### API connection fails
- Verify server is running
- Check device is on same network
- Test server URL in browser

## Next Steps

1. Review full documentation in README.md
2. Read USAGE.md for detailed user guide
3. Check BUILD_INSTRUCTIONS.md for advanced build options
4. Configure your API server
5. Test on physical device
6. Sign APK for production release

## Support Resources

- README.md - Complete documentation
- USAGE.md - User instructions
- BUILD_INSTRUCTIONS.md - Build details
- PROJECT_SUMMARY.md - Project overview

## Ready to Go!

Your Android app is ready. Just:
1. Open in Android Studio
2. Build
3. Run
4. Test with your API server

Good luck with your Food-101 recognition project!
