package com.example.foodrecognizer;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class ResultAdapter extends RecyclerView.Adapter<ResultAdapter.ResultViewHolder> {
    private List<Result> results;

    public ResultAdapter(List<Result> results) {
        this.results = results;
    }

    @NonNull
    @Override
    public ResultViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.result_item, parent, false);
        return new ResultViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ResultViewHolder holder, int position) {
        Result result = results.get(position);
        holder.dishName.setText(result.getDish());
        holder.confidence.setText(result.getConfidence() + "%");
        holder.rank.setText(String.valueOf(position + 1));
    }

    @Override
    public int getItemCount() {
        return results.size();
    }

    public void updateResults(List<Result> newResults) {
        this.results = newResults;
        notifyDataSetChanged();
    }

    static class ResultViewHolder extends RecyclerView.ViewHolder {
        TextView rank;
        TextView dishName;
        TextView confidence;

        ResultViewHolder(View itemView) {
            super(itemView);
            rank = itemView.findViewById(R.id.rank);
            dishName = itemView.findViewById(R.id.dishName);
            confidence = itemView.findViewById(R.id.confidence);
        }
    }
}
