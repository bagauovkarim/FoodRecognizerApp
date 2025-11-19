package com.example.foodrecognizer;

import android.graphics.Bitmap;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ApiClient {
    private static final String TAG = "ApiClient";
    private OkHttpClient client;
    private Gson gson;

    public ApiClient() {
        client = new OkHttpClient();
        gson = new Gson();
    }

    public interface ApiCallback {
        void onSuccess(List<Result> results);
        void onError(String error);
    }

    public void sendImage(Bitmap bitmap, String serverUrl, final ApiCallback callback) {
        // Convert bitmap to byte array
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream);
        byte[] byteArray = stream.toByteArray();

        // Create multipart request body
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "photo.jpg",
                        RequestBody.create(byteArray, MediaType.parse("image/jpeg")))
                .build();

        // Build request
        Request request = new Request.Builder()
                .url(serverUrl + "/predict")
                .post(requestBody)
                .build();

        // Execute request asynchronously
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(TAG, "Request failed", e);
                callback.onError("Network error: " + e.getMessage());
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    callback.onError("Server error: " + response.code());
                    return;
                }

                try {
                    String responseBody = response.body().string();
                    Log.d(TAG, "Response: " + responseBody);

                    // Parse JSON response
                    JsonObject jsonObject = gson.fromJson(responseBody, JsonObject.class);

                    if (jsonObject.has("success") && jsonObject.get("success").getAsBoolean()) {
                        List<Result> results = new ArrayList<>();
                        JsonArray predictions = jsonObject.getAsJsonArray("predictions");

                        for (int i = 0; i < predictions.size(); i++) {
                            JsonObject prediction = predictions.get(i).getAsJsonObject();
                            String dish = prediction.get("dish").getAsString();
                            String confidence = prediction.get("confidence").getAsString();
                            results.add(new Result(dish, confidence));
                        }

                        callback.onSuccess(results);
                    } else {
                        callback.onError("Prediction failed");
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error parsing response", e);
                    callback.onError("Error parsing response: " + e.getMessage());
                }
            }
        });
    }
}
