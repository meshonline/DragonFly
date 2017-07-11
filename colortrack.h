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
#include "familycluster.h"

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
// Use small screen with slow CPU.
#define QVGA false
// Star point threshold.
#define MAX_DISTANCE_THRESHOLD (QVGA ? 10 : 20)

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
    "COLOR_SIZE"
};

typedef struct {
    float mean_h;
    float mean_s;
    float mean_v;
    float standard_deviation_h;
    float standard_deviation_s;
    float standard_deviation_v;
} Color;

typedef vector<Color> ColorGroup;

// Star point.
typedef struct StarPoint {
    StarPoint(const int _x, const int _y, const float& _power = 0.0f) {
        x = _x;
        y = _y;
        power = _power;
    }
    // Position.
    int x;
    int y;
    // Glow power.
    float power;
} StarPoint;

class parallelTestBody
: public ParallelLoopBody  // Refer to OpenCV's official answer，construct a parallel loop body.
{
public:
    parallelTestBody(vector<vector<StarPoint> >& _src,
                     vector<float>& _max_f,
                     vector<int>& _max_i)  // class constructor
    {
        src = &_src;
        max_f = &_max_f;
        max_i = &_max_i;
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
            // Fast compute the distances of all combination of the star points.
            for (int i = 0; i < static_cast<int>((*src)[c].size()); i++) {
                for (int j = i + 1; j < static_cast<int>((*src)[c].size()); j++) {
                    int dx = (*src)[c][j].x - (*src)[c][i].x;
                    // Ignore too left and too right star points.
                    if (dx < -MAX_DISTANCE_THRESHOLD || dx > MAX_DISTANCE_THRESHOLD) {
                        continue;
                    }
                    
                    int dy = (*src)[c][j].y - (*src)[c][i].y;
                    // Ignore too lower star points.
                    if (dy > MAX_DISTANCE_THRESHOLD) {
                        break;
                    }
                    
                    // Calculate the distance.
                    float distance = CHESS_BOARD ? MAX(ABS(dx), ABS(dy))
                    : sqrtf((float)(dx * dx + dy * dy));
                    // Calculate the Glow power.
                    float power = 1.0f / distance;
                    // accumulate the glow power.
                    (*src)[c][i].power += power;
                    (*src)[c][j].power += power;
                }
                // Detect the brightest star, it must be in the center of the cluster with largest density.
                if ((*src)[c][i].power > (*max_f)[c]) {
                    (*max_f)[c] = (*src)[c][i].power;
                    (*max_i)[c] = i;
                }
            }
        }
    }
    vector<vector<StarPoint> >* src;
    vector<float>* max_f;
    vector<int>* max_i;
};

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

void mask_color_by_depth(Mat& mat_color,
                         const Mat& mat_depth,
                         const Mat& mat_background,
                         vector<Joint>& joints,
                         const ColorType learn_color = COLOR_SIZE,
                         int* px = NULL,
                         int* py = NULL) {
    // Get pointer from matrix.
    uint8_t* color_buffer = mat_color.ptr();
    const uint16_t* depth_buffer = (uint16_t*)mat_depth.ptr();
    const uint16_t* background_buffer = (uint16_t*)mat_background.ptr();
    
    // Array of color points.
    vector<vector<StarPoint> > points(JOINT_SIZE);
    vector<float> max_f(JOINT_SIZE, 0.0f);
    vector<int> max_i(JOINT_SIZE, 0);
    
    // Color table.
    static bool first_time = true;
    static vector<Vec3> color_table(COLOR_SIZE);
    static vector<ColorGroup> color_group_table(COLOR_SIZE);
    static vector<ColorType> joint_color(JOINT_SIZE);
    
    // Execute only once.
    if (first_time) {
        // Load the accurate color table generated by color learning.
        ifstream fin("hsv_learn.txt");
        string line;
        while(getline(fin, line))
        {
            string label;
            static int color_number = 0;
            static float mean_h = 0.0f;
            static float standard_deviation_h = 0.0f;
            static float mean_s = 0.0f;
            static float standard_deviation_s = 0.0f;
            static float mean_v = 0.0f;
            static float standard_deviation_v = 0.0f;
            
            static bool finished = false;

            if (line.find("color ") != string::npos) {
                stringstream ss(line);
                ss >> label >> color_number;
            }

            if (line.find("h ") != string::npos) {
                stringstream ss(line);
                ss >> label >> mean_h >> standard_deviation_h;
            }

            if (line.find("s ") != string::npos) {
                stringstream ss(line);
                ss >> label >> mean_s >> standard_deviation_s;
            }
            
            if (line.find("v ") != string::npos) {
                stringstream ss(line);
                ss >> label >> mean_v >> standard_deviation_v;
                finished = true;
            }
            
            if (finished) {
                Color one_color;
                one_color.mean_h = mean_h;
                one_color.mean_s = mean_s;
                one_color.mean_v = mean_v;
                one_color.standard_deviation_h = standard_deviation_h;
                one_color.standard_deviation_s = standard_deviation_s;
                one_color.standard_deviation_v = standard_deviation_v;
                color_group_table[color_number].push_back(one_color);
                finished = false;
            }
        }
        fin.close();

        // Fill color table.
        color_table[COLOR_BLACK] = {30.0f, 0.14f, 0.05f};
        color_table[COLOR_GRAY] = {340.0f, 0.08f, 0.45f};
        color_table[COLOR_WHITE] = {294.0f, 0.05f, 0.82f};
        color_table[COLOR_ORANGE] = {11.0f, 0.89f, 0.74f};
        color_table[COLOR_YELLOW] = {39.0f, 0.75f, 0.92f};
        color_table[COLOR_LIGHT_GREEN] = {80.27f, 0.87f, 0.67f};
        color_table[COLOR_GREEN] = {131.0f, 0.56f, 0.34f};
        color_table[COLOR_LIGHT_BLUE] = {207.0f, 0.43f, 0.66f};
        color_table[COLOR_DARK_BLUE] = {243.0f, 0.71f, 0.29f};
        color_table[COLOR_LIGHT_PURPLE] = {281.0f, 0.35f, 0.66f};
        color_table[COLOR_LIGHT_PINK] = {335.0f, 0.46f, 0.88f};
        color_table[COLOR_DARK_RED] = {339.0f, 0.92f, 0.52f};
        color_table[COLOR_RED] = {358.0f, 0.93f, 0.55f};
        
        // Assign color to each joint.
        joint_color[JOINT_HEAD] = COLOR_BLACK;
        joint_color[JOINT_NECK] = COLOR_SIZE;
        joint_color[JOINT_LEFT_SHOULDER] = COLOR_GRAY;
        joint_color[JOINT_RIGHT_SHOULDER] = COLOR_WHITE;
        joint_color[JOINT_LEFT_ELBOW] = COLOR_ORANGE;
        joint_color[JOINT_RIGHT_ELBOW] = COLOR_YELLOW;
        joint_color[JOINT_LEFT_HAND] = COLOR_LIGHT_GREEN;
        joint_color[JOINT_RIGHT_HAND] = COLOR_GREEN;
        joint_color[JOINT_TORSO] = COLOR_SIZE;
        joint_color[JOINT_LEFT_HIP] = COLOR_LIGHT_BLUE;
        joint_color[JOINT_RIGHT_HIP] = COLOR_DARK_BLUE;
        joint_color[JOINT_LEFT_KNEE] = COLOR_LIGHT_PURPLE;
        joint_color[JOINT_RIGHT_KNEE] = COLOR_LIGHT_PINK;
        joint_color[JOINT_LEFT_FOOT] = COLOR_DARK_RED;
        joint_color[JOINT_RIGHT_FOOT] = COLOR_RED;
        
        first_time = false;
    }
    
    for (int i = 0; i < (QVGA ? 240 : 480) * (QVGA ? 320 : 640); i++) {
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
                if (learn_color != COLOR_SIZE && joint_color[c] != learn_color) {
                    continue;
                }
                
                bool color_detected = false;
                
                // If exists the accurate color table generated by color learning.
                if (color_group_table[joint_color[c]].size()) {
                    for (int j=0; j<static_cast<int>(color_group_table[joint_color[c]].size()); j++) {
                        // If the color match the color table.
                        if (hue_difference(h, color_group_table[joint_color[c]][j].mean_h, joint_color[c]) <
                            color_group_table[joint_color[c]][j].standard_deviation_h * 3.0f &&
                            ABS(s - color_group_table[joint_color[c]][j].mean_s) <
                            color_group_table[joint_color[c]][j].standard_deviation_s * 3.0f &&
                            ABS(v - color_group_table[joint_color[c]][j].mean_v) <
                            color_group_table[joint_color[c]][j].standard_deviation_v * 3.0f) {
                            color_detected = true;
                            // We only need the match one of the color item in the color table.
                            break;
                        }
                    }
                } else {
                    // If the color match the color table.
                    if (hue_difference(h, color_table[joint_color[c]].x, joint_color[c]) < 5.8f &&
                        ABS(s - color_table[joint_color[c]].y) < 0.16f &&
                        ABS(v - color_table[joint_color[c]].z) < 0.25f) {
                        color_detected = true;
                    }
                }
                
                // If detected the color.
                if (color_detected) {
                    // Mark the color with red.
                    if (SHOW_JOINT_COLOR) {
                        color_buffer[index1] = 0;
                        color_buffer[index2] = 0;
                        color_buffer[index3] = 255;
                    }
                    
                    // Add the star point.
                    points[c].push_back(StarPoint(i % (QVGA ? 320 : 640), i / (QVGA ? 320 : 640)));
                }
            }
        }
    }
    
    if (PARALLEL_COMPUTE) {
        parallel_for_(Range(0, static_cast<int>(points.size())),
                      parallelTestBody(points, max_f, max_i));
    } else {
        parallel_for(BlockedRange(0, static_cast<int>(points.size())),
                     parallelTestBody(points, max_f, max_i));
    }
    
    // Loop through all colors in the color table.
    for (int c = 0; c < static_cast<int>(points.size()); c++) {
        // Ignore the joints with no color.
        if (joint_color[c] == COLOR_SIZE) {
            continue;
        }
        
        // If detected the color.
        if (points[c].size()) {
            // Calculate the space position of the center point.
            int cx = points[c][max_i[c]].x;
            int cy = points[c][max_i[c]].y;
            int wz = depth_buffer[cx + cy * (QVGA ? 320 : 640)];
            double wx, wy;
            freenect_sync_camera_to_world(QVGA ? cx * 2 : cx, QVGA ? cy * 2 : cy, wz,
                                          &wx, &wy, 0);
            
            // Update the position, and convert to OpenGL's right hand coordinate.
            joints[c].pos.x = wx;
            joints[c].pos.y = -wy;
            joints[c].pos.z = -wz;
            // Update the tracking status.
            if (max_f[c] > CONFIDENCE_THRESHOLD) {
                joints[c].tracked = true;
            }
            
            // Draw circle.
            if (HIDE_WEAK_COLOR) {
                if (max_f[c] > CONFIDENCE_THRESHOLD) {
                    circle(mat_color, Point(points[c][max_i[c]].x, points[c][max_i[c]].y),
                           MAX_DISTANCE_THRESHOLD,
                           max_f[c] > 10.0f ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);
                }
            } else {
                circle(mat_color, Point(points[c][max_i[c]].x, points[c][max_i[c]].y),
                       MAX_DISTANCE_THRESHOLD,
                       max_f[c] > 10.0f ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);
            }

            // Output the coordinate of the leaning color.
            if (joint_color[c] == learn_color && max_f[c] > CONFIDENCE_THRESHOLD) {
                *px = points[c][max_i[c]].x;
                *py = points[c][max_i[c]].y;
            }
        }
    }
}

void do_color_learning(const vector<Vec3>& color_list, Vec3& mean, Vec3& standard_deviation) {
    // Accumulate three channels(HSV).
    for (int c=0; c<static_cast<int>(color_list.size()); c++) {
        mean.x += color_list[c].x;
        mean.y += color_list[c].y;
        mean.z += color_list[c].z;
    }

    // Calculate the mean value of three channels.
    mean.x /= color_list.size();
    mean.y /= color_list.size();
    mean.z /= color_list.size();

    // Calculate the variances of three channels.
    for (int c=0; c<static_cast<int>(color_list.size()); c++) {
        standard_deviation.x += (color_list[c].x - mean.x) * (color_list[c].x - mean.x);
        standard_deviation.y += (color_list[c].y - mean.y) * (color_list[c].y - mean.y);
        standard_deviation.z += (color_list[c].z - mean.z) * (color_list[c].z - mean.z);
    }

    // Calculate the average variances of three channels.
    standard_deviation.x /= color_list.size();
    standard_deviation.y /= color_list.size();
    standard_deviation.z /= color_list.size();

    // Calculate the standard deviations of three channels.
    standard_deviation.x = sqrtf(standard_deviation.x);
    standard_deviation.y = sqrtf(standard_deviation.y);
    standard_deviation.z = sqrtf(standard_deviation.z);
}

void machine_learning(const vector<Vec3>& color_list,
                      const float& rate_tolerance,
                      const float& amount_tolerance,
                      const ColorType learn_color,
                      const string& filename) {
    float* data = new float[color_list.size() * 3];
    float* normalized_color = new float[color_list.size() * 3];
    float* distance = new float[color_list.size() * color_list.size()];
    float* rate = new float[color_list.size() - 1];
    int* entity = new int[(color_list.size() - 1) * 2];
    for (int i = 0; i < static_cast<int>(color_list.size()); i++) {
        data[i * 3 + 0] = color_list[i].x;
        data[i * 3 + 1] = color_list[i].y;
        data[i * 3 + 2] = color_list[i].z;
    }
    make_normalization(data, static_cast<int>(color_list.size()), 3,
                       normalized_color);
    make_distance(normalized_color, static_cast<int>(color_list.size()),
                  3, distance);
    do_cluster(distance, static_cast<int>(color_list.size()), rate,
               entity);
    // Cluster by family.
    set<int>** family = new set<int>*[color_list.size()];
    set<int>** origin_family = new set<int>*[color_list.size()];
    bool* family_alive = new bool[color_list.size()];
    // Initial
    for (int i = 0; i < color_list.size(); i++) {
        family[i] = new set<int>;
        family[i]->insert(i);
        family_alive[i] = true;
        // Save pointers for cleaning.
        origin_family[i] = family[i];
    }
    // Start to cluster.
    for (int i = 0; i < color_list.size() - 1; i++) {
        // If reach to rate threshold, finish the clustering.
        if (rate[i] > rate_tolerance) {
            break;
        }
        // Combine the second family to the first family.
        family[entity[i * 2]]->insert(family[entity[i * 2 + 1]]->begin(),
                                      family[entity[i * 2 + 1]]->end());
        // Loop through the first family.
        for (set<int>::const_iterator it = family[entity[i * 2]]->begin();
             it != family[entity[i * 2]]->end(); it++) {
            // Let every family point to the first family.
            family[*it] = family[entity[i * 2]];
            // Mark every family as unalive.
            family_alive[*it] = false;
        }
        // Mark the first family as alive.
        family_alive[entity[i * 2]] = true;
    }
    for (int i = 0; i < color_list.size(); i++) {
        if (family_alive[i]) {
            // Ignore the rare family.
            if (family[i]->size() > color_list.size() * amount_tolerance) {
                vector<Vec3> one_color_list;
                for (set<int>::const_iterator it = family[i]->begin();
                     it != family[i]->end(); it++) {
                    one_color_list.push_back(color_list[*it]);
                }
                
                Vec3 one_mean = vec3_zero;
                Vec3 one_standard_deviation = vec3_zero;
                // calculate the mean and the standard deviation for the family.
                do_color_learning(one_color_list, one_mean, one_standard_deviation);
                // Save the result to file.
                time_t nowtime = time(NULL);
                struct tm* local = localtime(&nowtime);
                char buf[256];
                sprintf(buf, "%d-%d-%d-%d-%d-%d",
                        local->tm_year + 1900, local->tm_mon + 1, local->tm_mday,
                        local->tm_hour, local->tm_min, local->tm_sec);
                ofstream of(filename, ios::app);
                of << "time " << buf << endl;
                of << "size " << family[i]->size() << endl;
                of << "color " << learn_color << endl;
                of << "h " << one_mean.x << " " << one_standard_deviation.x << endl;
                of << "s " << one_mean.y << " " << one_standard_deviation.y << endl;
                of << "v " << one_mean.z << " " << one_standard_deviation.z << endl
                << endl;
                of.close();
            }
        }
    }
    // Clean the families.
    for (int i = 0; i < color_list.size(); i++) {
        delete origin_family[i];
    }
    // Clean other data.
    delete [] family;
    delete [] origin_family;
    delete [] family_alive;
    delete[] data;
    delete[] normalized_color;
    delete[] distance;
    delete[] rate;
    delete[] entity;
}

#endif /* colortrack_h */
