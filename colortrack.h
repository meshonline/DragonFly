//
//  colortrack.h
//  DragonFly
//
//  Created by MINGFENWANG on 2017/6/18.
//  Copyright (c) 2017 MINGFENWANG. All rights reserved.
//

#ifndef colortrack_h
#define colortrack_h

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
#include "kinectbvh.h"

#define ABS(a) (((a) > 0) ? (a) : (-(a)))

// Chess board distance or Euler distance.
#define CHESS_BOARD true
// Use multi core CPU to parallel compute.
#define PARALLEL_COMPUTE true
// Mark joint pixels with color.
#define SHOW_JOINT_COLOR false
// Hide joint with low confidence.
#define HIDE_WEAK_COLOR true
// Confidence threshold
#define CONFIDENCE_THRESHOLD 10
// The cut distance before the background.
#define CUT_DEPTH 150
// Star point threshold.
#define MAX_DISTANCE_THRESHOLD 20

using namespace cv;
using namespace std;

typedef enum {
    COLOR_BLACK,
    COLOR_GRAY,
    COLOR_WHITE,
    COLOR_ORANGE,
    COLOR_YELLOW,
    COLOR_LIGHT_GREEN,
    COLOR_GREEN,
    COLOR_LIGHT_BLUE,
    COLOR_DARK_BLUE,
    COLOR_LIGHT_PURPLE,
    COLOR_LIGHT_PINK,
    COLOR_DARK_RED,
    COLOR_RED,
    COLOR_BROWN,
    COLOR_SIZE
} ColorType;

static const char* ColorName[] = {
    "COLOR_BLACK",
    "COLOR_GRAY",
    "COLOR_WHITE",
    "COLOR_ORANGE",
    "COLOR_YELLOW",
    "COLOR_LIGHT_GREEN",
    "COLOR_GREEN",
    "COLOR_LIGHT_BLUE",
    "COLOR_DARK_BLUE",
    "COLOR_LIGHT_PURPLE",
    "COLOR_LIGHT_PINK",
    "COLOR_DARK_RED",
    "COLOR_RED",
    "COLOR_BROWN",
    "COLOR_SIZE"
};

// Star point.
typedef struct StarPoint {
    StarPoint(const int _x = 0, const int _y = 0, const int _z = 0) {
        x = _x;
        y = _y;
        z = _z;
    }
    // Position.
    int16_t x;
    int16_t y;
    int16_t z;
} StarPoint;

class parallelTestBody
: public ParallelLoopBody  // Refer to OpenCV's official answer，construct a parallel loop body.
{
public:
    parallelTestBody(vector<vector<StarPoint> >& _points,
                     vector<float>& _max_power,
                     vector<int>& _max_index)  // class constructor
    {
        points = &_points;
        max_power = &_max_power;
        max_index = &_max_index;
    }
    void operator()(const BlockedRange& range)
    const  // To use BlockedRange class, we need to include opencv2/core/internal.hpp.
    {
        compute_all(range.begin(), range.end());
    }
    void operator()(const Range& range) const  // override the operator().
    {
        compute_all(range.start, range.end);
    }
    
private:
    void compute_all(const int start, const int end) const {
        // Loop through the colors in the range.
        for (int c = start; c < end; c++) {
            // Powers array
            vector<float> powers((*points)[c].size(), 0.0f);
            // Fast compute the distances of all combination of the star points.
            for (int i = 0; i < static_cast<int>((*points)[c].size()); i++) {
                for (int j = i + 1; j < static_cast<int>((*points)[c].size()); j++) {
                    int dx = (*points)[c][j].x - (*points)[c][i].x;
                    // Ignore too left and too right star points.
                    if (dx < -MAX_DISTANCE_THRESHOLD || dx > MAX_DISTANCE_THRESHOLD) {
                        continue;
                    }
                    
                    int dy = (*points)[c][j].y - (*points)[c][i].y;
                    // Ignore too lower star points.
                    if (dy > MAX_DISTANCE_THRESHOLD) {
                        break;
                    }
                    
                    // Calculate the distance.
                    float distance = CHESS_BOARD ? MAX(ABS(dx), ABS(dy))
                    : sqrtf((float)(dx * dx + dy * dy));
                    // Calculate the glow power.
                    float power = 1.0f / distance;
                    // Accumulate the glow power.
                    powers[i] += power;
                    powers[j] += power;
                }
                // Detect the brightest star, it must be in the center of the cluster with largest density.
                if (powers[i] > (*max_power)[c]) {
                    (*max_power)[c] = powers[i];
                    (*max_index)[c] = i;
                }
            }
        }
    }
    vector<vector<StarPoint> >* points;
    vector<float>* max_power;
    vector<int>* max_index;
};

class Color_Tracker {
public:
    Color_Tracker() {
        // Resize color table, color group table and joint color table.
        color_table.resize(COLOR_SIZE);
        joint_color.resize(JOINT_SIZE);

        // Fill color table.
        color_table[COLOR_BLACK] = {150.0f, 0.6f, 0.04f};
        color_table[COLOR_GRAY] = {60.0f, 0.2f, 0.36f};
        color_table[COLOR_WHITE] = {227.0f, 0.04f, 0.93f};
        color_table[COLOR_ORANGE] = {12.0f, 0.88f, 0.75f};
        color_table[COLOR_YELLOW] = {40.0f, 0.74f, 0.99f};
        color_table[COLOR_LIGHT_GREEN] = {81.0f, 0.78f, 0.75f};
        color_table[COLOR_GREEN] = {138.0f, 0.53f, 0.38f};
        color_table[COLOR_LIGHT_BLUE] = {211.0f, 0.55f, 0.66f};
        color_table[COLOR_DARK_BLUE] = {232.0f, 0.76f, 0.34f};
        color_table[COLOR_LIGHT_PURPLE] = {281.0f, 0.36f, 0.56f};
        color_table[COLOR_LIGHT_PINK] = {336.0f, 0.4f, 0.83f};
        color_table[COLOR_DARK_RED] = {340.0f, 0.84f, 0.43f};
        color_table[COLOR_RED] = {357.0f, 0.85f, 0.43f};
        color_table[COLOR_BROWN] = {25.0f, 0.69f, 0.36f};
        
        // Assign color to each joint.
        joint_color[JOINT_HEAD] = COLOR_BLACK;
        joint_color[JOINT_NECK] = COLOR_SIZE;
        joint_color[JOINT_LEFT_SHOULDER] = COLOR_GRAY;
        joint_color[JOINT_RIGHT_SHOULDER] = COLOR_YELLOW;
        joint_color[JOINT_LEFT_ELBOW] = COLOR_ORANGE;
        joint_color[JOINT_RIGHT_ELBOW] = COLOR_BROWN;
        joint_color[JOINT_LEFT_HAND] = COLOR_LIGHT_GREEN;
        joint_color[JOINT_RIGHT_HAND] = COLOR_GREEN;
        joint_color[JOINT_TORSO] = COLOR_SIZE;
        joint_color[JOINT_LEFT_HIP] = COLOR_LIGHT_BLUE;
        joint_color[JOINT_RIGHT_HIP] = COLOR_DARK_BLUE;
        joint_color[JOINT_LEFT_KNEE] = COLOR_LIGHT_PURPLE;
        joint_color[JOINT_RIGHT_KNEE] = COLOR_LIGHT_PINK;
        joint_color[JOINT_LEFT_FOOT] = COLOR_DARK_RED;
        joint_color[JOINT_RIGHT_FOOT] = COLOR_RED;
    }

    bool get_learn_hsv(Mat& mat_color,
                             const Mat& mat_depth,
                             const Mat& mat_background,
                             const ColorType learn_color,
                             Vec3* hsv) {
        // Get pointer from matrix.
        uint8_t* color_buffer = mat_color.ptr();
        const uint16_t* depth_buffer = (uint16_t*)mat_depth.ptr();
        const uint16_t* background_buffer = (uint16_t*)mat_background.ptr();
        
        // Array of color points.
        vector<vector<StarPoint> > points(JOINT_SIZE);
        vector<float> max_power(JOINT_SIZE, 0.0f);
        vector<int> max_index(JOINT_SIZE, 0);
        
        for (int i = 0; i < 480 * 640; i++) {
            int index1 = i + i + i;
            int index2 = index1 + 1;
            int index3 = index2 + 1;
            // Is the pixel in the background ?
            bool is_background = (depth_buffer[i] == 0 ||
                                  depth_buffer[i] + CUT_DEPTH > background_buffer[i]);
            // If it is in the background.
            if (is_background) {
                // Clear the color.
                color_buffer[index1] = color_buffer[index2] = color_buffer[index3] = 0;
            } else {
                // Convert to HSV format.
                float h, s, v;
                bgr_to_hsv(color_buffer[index1], color_buffer[index2], color_buffer[index3], h, s, v);
                
                // Loop through all colors in the color table.
                for (int c = 0; c < static_cast<int>(points.size()); c++) {
                    // Ignore the joints with no color.
                    if (joint_color[c] == COLOR_SIZE) {
                        continue;
                    }
                    
                    // Ignore the joints which are not the current learning color.
                    if (joint_color[c] != learn_color) {
                        continue;
                    }
                    
                    bool color_detected = false;
                    
                    // If the color match the color table.
                    if (hue_difference(h, color_table[joint_color[c]].x, joint_color[c]) < 5.8f &&
                        ABS(s - color_table[joint_color[c]].y) < 0.16f &&
                        ABS(v - color_table[joint_color[c]].z) < 0.25f) {
                        color_detected = true;
                    }
                    
                    // If color detected.
                    if (color_detected) {
                        // Mark the color with red.
                        if (SHOW_JOINT_COLOR) {
                            color_buffer[index1] = 0;
                            color_buffer[index2] = 0;
                            color_buffer[index3] = 255;
                        }
                        
                        // Add the star point.
                        int x = i % 640;
                        int y = i / 640;
                        int z = depth_buffer[x + y * 640];
                        points[c].push_back(StarPoint(x, y, z));
                    }
                }
            }
        }

        if (PARALLEL_COMPUTE) {
            parallel_for_(Range(0, static_cast<int>(points.size())),
                          parallelTestBody(points, max_power, max_index));
        } else {
            parallel_for(BlockedRange(0, static_cast<int>(points.size())),
                         parallelTestBody(points, max_power, max_index));
        }
        
        bool success = false;
        // Loop through all colors in the color table.
        for (int c = 0; c < static_cast<int>(points.size()); c++) {
            // Ignore the joints with no color.
            if (joint_color[c] == COLOR_SIZE) {
                continue;
            }
            
            // Ignore the joints which are not the current learning color.
            if (joint_color[c] != learn_color) {
                continue;
            }
            
            // If color detected.
            if (points[c].size()) {
                // Draw circle.
                if (!HIDE_WEAK_COLOR) {
                    // Red or green circle.
                    circle(mat_color, Point(points[c][max_index[c]].x, points[c][max_index[c]].y),
                           MAX_DISTANCE_THRESHOLD,
                           max_power[c] > CONFIDENCE_THRESHOLD ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);
                } else if (max_power[c] > CONFIDENCE_THRESHOLD) {
                    // Red circle.
                    circle(mat_color, Point(points[c][max_index[c]].x, points[c][max_index[c]].y),
                           MAX_DISTANCE_THRESHOLD, Scalar(0, 0, 255), 2);
                }
                
                // Output hsv of the leaning color.
                if (max_power[c] > CONFIDENCE_THRESHOLD) {
                    int x = points[c][max_index[c]].x;
                    int y = points[c][max_index[c]].y;
                    uchar* color_ptr = mat_color.ptr(y) + 3 * x;
                    bgr_to_hsv(color_ptr[0], color_ptr[1], color_ptr[2], hsv->x, hsv->y, hsv->z);
                    success = true;
                }
            }
        }
        return success;
    }
    
    void generate_star_points(Mat& mat_color,
                              const Mat& mat_depth,
                              const Mat& mat_background,
                              ofstream& fout) {
        // Get pointer from matrix.
        uint8_t* color_buffer = mat_color.ptr();
        const uint16_t* depth_buffer = (uint16_t*)mat_depth.ptr();
        const uint16_t* background_buffer = (uint16_t*)mat_background.ptr();
        
        // Array of color points.
        vector<vector<StarPoint> > points(JOINT_SIZE);
        vector<float> max_power(JOINT_SIZE, 0.0f);
        vector<int> max_index(JOINT_SIZE, 0);
        
        for (int i = 0; i < 480 * 640; i++) {
            int index1 = i + i + i;
            int index2 = index1 + 1;
            int index3 = index2 + 1;
            // Is the pixel in the background ?
            bool is_background = (depth_buffer[i] == 0 ||
                                  depth_buffer[i] + CUT_DEPTH > background_buffer[i]);
            // If it is in the background.
            if (is_background) {
                // Clear the color.
                color_buffer[index1] = color_buffer[index2] = color_buffer[index3] = 0;
            } else {
                // Convert to HSV format.
                float h, s, v;
                bgr_to_hsv(color_buffer[index1], color_buffer[index2], color_buffer[index3], h, s, v);
                
                // Loop through all colors in the color table.
                for (int c = 0; c < static_cast<int>(points.size()); c++) {
                    // Ignore the joints with no color.
                    if (joint_color[c] == COLOR_SIZE) {
                        continue;
                    }
                    
                    bool color_detected = false;
                    
                    // If the color match the color table.
                    if (hue_difference(h, color_table[joint_color[c]].x, joint_color[c]) < 5.8f &&
                        ABS(s - color_table[joint_color[c]].y) < 0.16f &&
                        ABS(v - color_table[joint_color[c]].z) < 0.25f) {
                        color_detected = true;
                    }
                    
                    // If color detected.
                    if (color_detected) {
                        // Mark the color with red.
                        if (SHOW_JOINT_COLOR) {
                            color_buffer[index1] = 0;
                            color_buffer[index2] = 0;
                            color_buffer[index3] = 255;
                        }
                        
                        // Add the star point.
                        int x = i % 640;
                        int y = i / 640;
                        int z = depth_buffer[x + y * 640];
                        points[c].push_back(StarPoint(x, y, z));
                    }
                }
            }
        }
        
        // Loop through all colors in the color table.
        for (int c = 0; c < static_cast<int>(points.size()); c++) {
            int star_count = static_cast<int>(points[c].size());
            fout.write((const char*)&star_count, sizeof(int));
            if (star_count > 0) {
                fout.write((const char*)&points[c][0], sizeof(StarPoint) * star_count);
            }
        }
    }
    
    void process_star_points(vector<Joint>& joints,
                              ifstream& fin) {
        // Array of color points.
        vector<vector<StarPoint> > points(JOINT_SIZE);
        vector<float> max_power(JOINT_SIZE, 0.0f);
        vector<int> max_index(JOINT_SIZE, 0);
        
        // Loop through all colors in the color table.
        for (int c = 0; c < static_cast<int>(points.size()); c++) {
            int star_count;
            fin.read((char*)&star_count, sizeof(int));
            points[c].resize(star_count);
            if (star_count > 0) {
                fin.read((char*)&points[c][0], sizeof(StarPoint) * star_count);
            }
        }
        
        if (PARALLEL_COMPUTE) {
            parallel_for_(Range(0, static_cast<int>(points.size())),
                          parallelTestBody(points, max_power, max_index));
        } else {
            parallel_for(BlockedRange(0, static_cast<int>(points.size())),
                         parallelTestBody(points, max_power, max_index));
        }
        
        // Loop through all colors in the color table.
        for (int c = 0; c < static_cast<int>(points.size()); c++) {
            // Ignore the joints with no color.
            if (joint_color[c] == COLOR_SIZE) {
                continue;
            }
            
            // If color detected.
            if (points[c].size()) {
                // Calculate the space position of the center point.
                int cx = points[c][max_index[c]].x;
                int cy = points[c][max_index[c]].y;
                int wz = points[c][max_index[c]].z;
                double wx, wy;
                freenect_sync_camera_to_world(cx, cy, wz, &wx, &wy, 0);
                
                // Update the position, and convert to OpenGL's right hand coordinate.
                joints[c].pos.x = wx;
                joints[c].pos.y = -wy;
                joints[c].pos.z = -wz;
                // Update the tracking status.
                if (max_power[c] > CONFIDENCE_THRESHOLD) {
                    joints[c].tracked = true;
                }
            }
        }
    }
    
    inline void bgr_to_hsv(const uint8_t b,
                           const uint8_t g,
                           const uint8_t r,
                           float& h,
                           float& s,
                           float& v) {
        uint8_t max = MAX(b, MAX(g, r));
        uint8_t min = MIN(b, MIN(g, r));
        // Not defined, but we treat it as washed red color.
        if (max == min) {
            h = s = 0.0f;
        } else {
            float delta = (float)(max - min);
            if (r == max) {
                h = (float)(g - b) / delta;
            }
            if (g == max) {
                h = 2.0f + (float)(b - r) / delta;
            }
            if (b == max) {
                h = 4.0f + (float)(r - g) / delta;
            }
            h *= 60.0f;
            if (h < 0.0f) {
                h += 360.0f;
            }
            // The bigger the different of the channels, the bigger the satuation.
            s = (float)delta / max;
        }
        // The color value is the value of the brightest channel.
        v = (float)max / 255.0f;
    }
    
    inline void fast_bgr_to_hsv(const uint8_t b,
                                const uint8_t g,
                                const uint8_t r,
                                float& h,
                                float& s,
                                float& v) {
        uint8_t max = MAX(b, MAX(g, r));
        uint8_t min = MIN(b, MIN(g, r));
        // Not defined, but we treat it as washed red color.
        if (max == min) {
            h = s = 0.0f;
        } else {
            // Rotate the hue by counter clockwise.
            h = (float)(max - r + g - min + b - min) / (max - min) * 60.0f;
            // If blue is brighter than green，then we rotate the hue by clockwise.
            if (g < b) {
                h = 360.0f - h;
            }
            // The bigger the different of the channels, the bigger the satuation.
            s = 1.0f - (float)min / max;
        }
        // The color value is the value of the brightest channel.
        v = (float)max / 255.0f;
    }
    
private:
    // http://blog.csdn.net/cay22/archive/2010/04/28/5535245.aspx
    ///////////////////////////////////////////////////////////////////////////////////////////
    // L = Y1 = (13933 * R + 46871 * G + 4732 * B) div 2^16
    // a = 377 * (14503 * R - 22218 * G + 7714 * B) div 2^24 + 128
    // b = 160 * (12773 * R + 39695 * G - 52468 * B) div 2^24 + 128
    inline void RGB2Lab2(const uint8_t R,
                         const uint8_t G,
                         const uint8_t B,
                         uint8_t& L,
                         uint8_t& a,
                         uint8_t& b) {
        L = 0.2126007 * R + 0.7151947 * G + 0.0722046 * B;
        a = 0.3258962 * R - 0.4992596 * G + 0.1733409 * B + 128;
        b = 0.1218128 * R + 0.3785610 * G - 0.5003738 * B + 128;
    }
    
    // R = L1 + (a1 * 100922 + b1 * 17790) div 2^23
    // G = L1 - (a1 * 30176 + b1 * 1481) div 2^23
    // B = L1 + (a1 * 1740 - b1 * 37719) div 2^23
    inline void Lab2RGB2(const uint8_t L,
                         const uint8_t a,
                         const uint8_t b,
                         uint8_t& R,
                         uint8_t& G,
                         uint8_t& B) {
        R = L + 0.0120308 * a + 0.0021207 * b;
        G = L - 0.0035973 * a - 0.0001765 * b;
        B = L + 0.0002074 * a - 0.0044965 * b;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    inline void normalize_color(uint8_t& b, uint8_t& g, uint8_t& r) {
        uint8_t max = MAX(b, MAX(g, r));
        if (max > 0) {
            b = b * 255 / max;
            g = g * 255 / max;
            r = r * 255 / max;
        }
    }
    
    inline float hue_difference(const float& h1, const float& h2, const ColorType learn_color) {
        if (learn_color == COLOR_BLACK || learn_color == COLOR_GRAY || learn_color == COLOR_WHITE) {
            return 0.0f;
        }
        float difference = ABS(h1 - h2);
        return (difference < 180.0f) ? difference : 360.0f - difference;
    }
    
    vector<Vec3> color_table;
    vector<ColorType> joint_color;
};

#endif /* colortrack_h */
