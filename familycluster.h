//
//  familycluster.h
//  DragonFly
//
//  Created by MINGFENWANG on 2017/6/27.
//  Copyright (c) 2017 MINGFENWANG. All rights reserved.
//

#ifndef familycluster_h
#define familycluster_h

#define ABS(a) (((a) > 0) ? (a) : (-(a)))

// Normalize the data.
void make_normalization(const float* data,
                        const int row,
                        const int column,
                        float* normalization) {
    for (int j = 0; j < column; j++) {
        float max = data[j];
        float min = data[j];
        for (int i = 0; i < row; i++) {
            if (data[i * column + j] > max) {
                max = data[i * column + j];
            }
            if (data[i * column + j] < min) {
                min = data[i * column + j];
            }
        }
        for (int i = 0; i < row; i++) {
            normalization[i * column + j] =
            (data[i * column + j] - min) / (max - min);
        }
    }
}

// Calculate the distance matrix.
void make_distance(const float* normalization,
                   const int row,
                   const int column,
                   float* distance) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < i; j++) {
            distance[i * row + j] = 0.0f;
            for (int k = 0; k < column; k++) {
                distance[i * row + j] +=
                ABS(normalization[i * column + k] - normalization[j * column + k]);
            }
        }
    }
}

// Direct cluster by distance.
void do_cluster(float* distance, const int row, float* rate, int* entity) {
    for (int n = 0; n < row - 1; n++) {
        // Temp variant to store the row and the column of the smallest distance.
        int min_row = 0;
        int min_column = 0;
        // Initial to maximum value.
        float min_value = FLT_MAX;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < i; j++) {
                // Detect the smallest distance.
                if (distance[i * row + j] < min_value) {
                    min_value = distance[i * row + j];
                    min_row = i;
                    min_column = j;
                }
            }
        }
        
        // Record the distance
        rate[n] = min_value;
        // Record the row.
        entity[n * 2] = min_row;
        // Record the column.
        entity[n * 2 + 1] = min_column;
        
        // Mark the contents of the column to maximum.
        for (int i = min_row + 1; i < row; i++) {
            distance[i * row + min_row] = FLT_MAX;
        }
        // Mark the contents of the row to maximum.
        for (int j = 0; j < min_row; j++) {
            distance[min_row * row + j] = FLT_MAX;
        }
    }
}

#endif /* familycluster_h */
