# Food Recognizer App - Usage Guide

## Quick Start

### 1. First Launch
When you first open the app, you'll see:
- A placeholder image area
- Two buttons: "Camera" and "Gallery"
- A server URL input field
- An empty results section

### 2. Configure Server URL
Enter your API server address in the format:
```
http://192.168.1.100:5000
```

The app will remember this URL for future sessions.

### 3. Select an Image

#### Option A: Take a Photo
1. Tap the "Camera" button
2. Grant camera permission if prompted
3. Take a photo of your food
4. The photo will appear in the preview area

#### Option B: Choose from Gallery
1. Tap the "Gallery" button
2. Grant storage permission if prompted
3. Select a food image from your device
4. The image will appear in the preview area

### 4. Get Predictions
1. After selecting an image, the "Recognize Food" button will appear
2. Tap "Recognize Food"
3. Wait for the analysis (loading indicator will show)
4. View the top 5 predictions with confidence scores

## Understanding Results

Results are displayed as cards showing:
- **Rank**: Position (1-5) with colored badge
- **Dish Name**: The predicted food item
- **Confidence**: Percentage indicating prediction accuracy

Example:
```
1. Pizza - 85.5%
2. Burger - 72.3%
3. Pasta - 65.8%
4. Salad - 45.2%
5. Sushi - 32.1%
```

## Permissions

The app requires the following permissions:

### Camera Permission
- **Required for**: Taking photos
- **When requested**: When you tap the Camera button for the first time
- **Can deny**: Yes, but camera functionality won't work

### Storage Permission
- **Required for**: Selecting images from gallery
- **When requested**: When you tap the Gallery button for the first time
- **Can deny**: Yes, but gallery selection won't work

### Internet Permission
- **Required for**: Sending images to API server
- **When requested**: Automatically granted (no prompt)
- **Can deny**: No

## Troubleshooting

### "Network error" message
- Check that your device is connected to the same network as the API server
- Verify the server URL is correct
- Ensure the API server is running

### "Server error" message
- The API server returned an error
- Check server logs for details
- Verify the server is properly configured

### "Permission denied" message
- You denied camera or storage permission
- Go to Settings > Apps > Food Recognizer > Permissions
- Enable the required permission

### Camera not opening
1. Check that camera permission is granted
2. Restart the app
3. Verify your device has a working camera

### Gallery not opening
1. Check that storage permission is granted
2. Restart the app
3. Verify you have images in your gallery

## Tips for Best Results

1. **Good Lighting**: Take photos in well-lit conditions
2. **Clear View**: Ensure the food is clearly visible
3. **Close-up**: Get close enough to show details
4. **Single Item**: Best results with one main food item
5. **Stable Connection**: Use a stable network connection

## Server Setup

Your API server should:
1. Run on port 5000 (or configure as needed)
2. Accept POST requests at `/predict`
3. Handle multipart/form-data with field name `image`
4. Return JSON with success and predictions array

Example server response:
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

## Network Configuration

### Local Network (Same WiFi)
```
http://192.168.1.100:5000
```
Replace `192.168.1.100` with your server's local IP

### External Server
```
http://your-domain.com:5000
```

### HTTPS
The app supports HTTPS:
```
https://your-secure-domain.com
```

Note: The app allows cleartext HTTP traffic for development purposes.

## Privacy

- Images are sent to your configured API server only
- No data is stored on external servers
- Server URL is saved locally on your device
- Images are temporarily cached for display only

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your API server is working correctly
3. Check device permissions
4. Review app logs in Android Studio
