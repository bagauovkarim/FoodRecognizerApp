package com.example.foodrecognizer;

public class Result {
    private String dish;
    private String confidence;

    public Result(String dish, String confidence) {
        this.dish = dish;
        this.confidence = confidence;
    }

    public String getDish() {
        return dish;
    }

    public void setDish(String dish) {
        this.dish = dish;
    }

    public String getConfidence() {
        return confidence;
    }

    public void setConfidence(String confidence) {
        this.confidence = confidence;
    }
}
