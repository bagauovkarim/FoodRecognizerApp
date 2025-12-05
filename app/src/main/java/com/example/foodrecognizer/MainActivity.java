package com.example.foodrecognizer;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_IMAGE_PICK = 2;
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_STORAGE_PERMISSION = 101;
    private static final String PREFS_NAME = "FoodRecognizerPrefs";
    private static final String KEY_SERVER_URL = "server_url";

    private ImageView imagePreview;
    private EditText serverUrlInput;
    private Button btnCamera;
    private Button btnGallery;
    private Button btnPredict;
    private RecyclerView resultsRecyclerView;
    private ProgressBar progressBar;
    private TextView noResultsText;

    private Bitmap selectedImage;
    private ApiClient apiClient;
    private ResultAdapter resultAdapter;
    private Uri photoUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize views
        imagePreview = findViewById(R.id.imagePreview);
        serverUrlInput = findViewById(R.id.serverUrlInput);
        btnCamera = findViewById(R.id.btnCamera);
        btnGallery = findViewById(R.id.btnGallery);
        btnPredict = findViewById(R.id.btnPredict);
        resultsRecyclerView = findViewById(R.id.resultsRecyclerView);
        progressBar = findViewById(R.id.progressBar);
        noResultsText = findViewById(R.id.noResultsText);

        // Initialize API client
        apiClient = new ApiClient();

        // Setup RecyclerView
        resultAdapter = new ResultAdapter(new ArrayList<>());
        resultsRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        resultsRecyclerView.setAdapter(resultAdapter);

        // Load saved server URL
        loadServerUrl();

        // Set click listeners
        btnCamera.setOnClickListener(v -> openCamera());
        btnGallery.setOnClickListener(v -> openGallery());
        btnPredict.setOnClickListener(v -> predictImage());

        // Initially hide predict button
        btnPredict.setVisibility(View.GONE);
    }

    private void loadServerUrl() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String savedUrl = prefs.getString(KEY_SERVER_URL, "http://192.168.1.100:5000");
        serverUrlInput.setText(savedUrl);
    }

    private void saveServerUrl(String url) {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        prefs.edit().putString(KEY_SERVER_URL, url).apply();
    }

    private void openCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION);
            return;
        }

        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create file for full size image
            File photoFile = new File(getCacheDir(), "photo.jpg");
            photoUri = FileProvider.getUriForFile(this,
                    "com.example.foodrecognizer.fileprovider",
                    photoFile);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    private void openGallery() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_STORAGE_PERMISSION);
            return;
        }

        Intent pickPhoto = new Intent(Intent.ACTION_PICK,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickPhoto, REQUEST_IMAGE_PICK);
    }

    private void predictImage() {
        if (selectedImage == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        String serverUrl = serverUrlInput.getText().toString().trim();
        if (serverUrl.isEmpty()) {
            Toast.makeText(this, "Please enter server URL", Toast.LENGTH_SHORT).show();
            return;
        }

        // Save server URL
        saveServerUrl(serverUrl);

        // Show progress
        progressBar.setVisibility(View.VISIBLE);
        noResultsText.setVisibility(View.GONE);
        btnPredict.setEnabled(false);

        // Send image to API
        apiClient.sendImage(selectedImage, serverUrl, new ApiClient.ApiCallback() {
            @Override
            public void onSuccess(List<Result> results) {
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnPredict.setEnabled(true);

                    if (results.isEmpty()) {
                        noResultsText.setVisibility(View.VISIBLE);
                        noResultsText.setText("No results found");
                    } else {
                        resultAdapter.updateResults(results);
                        resultsRecyclerView.setVisibility(View.VISIBLE);
                    }

                    Toast.makeText(MainActivity.this,
                            "Prediction successful!", Toast.LENGTH_SHORT).show();
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnPredict.setEnabled(true);
                    noResultsText.setVisibility(View.VISIBLE);
                    noResultsText.setText("Error: " + error);
                    Toast.makeText(MainActivity.this,
                            "Error: " + error, Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            try {
                if (requestCode == REQUEST_IMAGE_CAPTURE) {
                    // Load image from file
                    selectedImage = MediaStore.Images.Media.getBitmap(
                            getContentResolver(), photoUri);
                    imagePreview.setImageBitmap(selectedImage);
                    btnPredict.setVisibility(View.VISIBLE);
                } else if (requestCode == REQUEST_IMAGE_PICK && data != null) {
                    Uri selectedImageUri = data.getData();
                    selectedImage = MediaStore.Images.Media.getBitmap(
                            getContentResolver(), selectedImageUri);
                    imagePreview.setImageBitmap(selectedImage);
                    btnPredict.setVisibility(View.VISIBLE);
                }

                // Clear previous results
                resultAdapter.updateResults(new ArrayList<>());
                noResultsText.setVisibility(View.GONE);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (requestCode == REQUEST_CAMERA_PERMISSION) {
                openCamera();
            } else if (requestCode == REQUEST_STORAGE_PERMISSION) {
                openGallery();
            }
        } else {
            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
        }
    }
}
