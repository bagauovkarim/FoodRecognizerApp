# Food Recognizer App

Native Android application for recognizing food using Food-101 AI model with real-time image classification.

## Features

- Take photos with camera or select from gallery
- Real-time food recognition using AI
- Top 5 predictions with confidence scores
- Beautiful Material Design UI
- Configurable API server URL
- Automatic server URL persistence

## Screenshots

The app provides a clean, modern interface with:
- Image preview
- Camera and gallery selection buttons
- Server URL configuration
- Top 5 prediction results with confidence percentages

## Requirements

- Android 7.0 (API 24) or higher
- Camera permission
- Internet permission
- Storage read permission

## API Server

The app connects to a Food-101 prediction API server that should:
- Accept POST requests at `/predict` endpoint
- Receive images as `multipart/form-data` with field name `image`
- Return JSON response in the following format:

```json
{
  "success": true,
  "predictions": [
    {"dish": "Pizza", "confidence": "85.5"},
    {"dish": "Burger", "confidence": "72.3"},
    {"dish": "Pasta", "confidence": "65.8"},
    {"dish": "Salad", "confidence": "45.2"},
    {"dish": "Sushi", "confidence": "32.1"}
  ]
}
```

## Installation

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd FoodRecognizerApp
```

2. Open the project in Android Studio

3. Build and run on your device or emulator

### From GitHub Actions

1. Go to the Actions tab in the repository
2. Download the latest APK artifact from successful builds
3. Install on your Android device

## Configuration

1. Launch the app
2. Enter your API server URL in the format: `http://YOUR_IP:5000`
3. Take a photo or select an image from gallery
4. Tap "Recognize Food" to get predictions

## Building

### Debug Build
```bash
./gradlew assembleDebug
```

### Release Build
```bash
./gradlew assembleRelease
```

## GitHub Actions CI/CD

This project includes automated builds via GitHub Actions:
- Automatic APK generation on push to main/master/develop branches
- Debug and Release APK artifacts available for download
- Test execution and reporting

### Workflow Features:
- Java 17 setup
- Gradle caching for faster builds
- APK artifact upload (30-day retention)
- Test results upload (7-day retention)

## Project Structure

```
FoodRecognizerApp/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── java/com/example/foodrecognizer/
│   │       │   ├── MainActivity.java          # Main activity
│   │       │   ├── ApiClient.java             # API communication
│   │       │   ├── Result.java                # Data model
│   │       │   └── ResultAdapter.java         # RecyclerView adapter
│   │       ├── res/
│   │       │   ├── layout/
│   │       │   │   ├── activity_main.xml      # Main UI layout
│   │       │   │   └── result_item.xml        # Result list item
│   │       │   ├── values/
│   │       │   │   ├── colors.xml             # Color definitions
│   │       │   │   ├── strings.xml            # String resources
│   │       │   │   └── themes.xml             # App themes
│   │       │   ├── drawable/
│   │       │   │   ├── rank_badge.xml         # Rank badge drawable
│   │       │   │   └── placeholder_image.xml  # Placeholder image
│   │       │   └── xml/
│   │       │       └── file_paths.xml         # FileProvider paths
│   │       └── AndroidManifest.xml            # App manifest
│   └── build.gradle                           # App build config
├── .github/
│   └── workflows/
│       └── android.yml                        # CI/CD workflow
├── build.gradle                               # Project build config
├── settings.gradle                            # Project settings
└── README.md                                  # This file
```

## Dependencies

- **AndroidX AppCompat**: Modern Android compatibility
- **Material Components**: Material Design UI components
- **ConstraintLayout**: Flexible layout system
- **RecyclerView**: Efficient list display
- **OkHttp**: HTTP client for API calls
- **Gson**: JSON parsing

## Permissions

The app requires the following permissions:
- `CAMERA`: To capture photos
- `READ_EXTERNAL_STORAGE`: To select images from gallery (API < 33)
- `READ_MEDIA_IMAGES`: To select images from gallery (API >= 33)
- `INTERNET`: To communicate with API server
- `ACCESS_NETWORK_STATE`: To check network connectivity

## Development

### Prerequisites
- Android Studio Arctic Fox or later
- JDK 17
- Android SDK 34
- Gradle 8.2

### Setup Development Environment
1. Install Android Studio
2. Install required SDK versions via SDK Manager
3. Clone the repository
4. Open project in Android Studio
5. Sync Gradle files
6. Run on emulator or device

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please open an issue in the repository.

## Author

Created for Food-101 recognition using deep learning AI models.
