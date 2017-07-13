//
//  main.cpp
//  DragonFly
//
//  Created by MINGFENWANG on 2017/6/15.
//  Copyright (c) 2017 MINGFENWANG. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
#include <libfreenect/libfreenect_sync.h>
#include "colortrack.h"
#include "kinectbvh.h"

int main(int argc, const char* argv[]) {
    // Prepare to grab data.
    void* data = NULL;
    uint32_t timestamp;
    
    // Make background buffer.
    uint16_t* background_buffer = new uint16_t[480 * 640];
    // Ignore unstable data.
    for (int i = 0; i < 30; i++) {
        freenect_sync_get_depth(&data, &timestamp, 0, FREENECT_DEPTH_REGISTERED);
    }
    // Copy to background buffer.
    memcpy(background_buffer, data, sizeof(uint16_t) * 480 * 640);
    // Pack background buffer to matrix.
    Mat mat_background(480, 640, CV_16UC1, background_buffer);
    // If using small screen, shrink the matrix.
    if (QVGA) {
        resize(mat_background, mat_background, Size(320, 240));
    }
    
    // Create a motion capture.
    KinectBVH* m_pKinectBVH = new KinectBVH();
    
    // Generate T pose skeleton.
    m_pKinectBVH->CalibrateSkeleton();
    
    bool snap_mode = false;
    int snap_counter = 150;
    bool learn_mode = false;
    int learn_color = COLOR_SIZE / 2;
    vector<Vec3> color_list;
    while (true) {
        int key = waitKey(1);
        
        // Press 'ESC' to quit.
        if (key == 27) {
            break;
        }
        // Press ‘l’ to switch to color learning mode.
        if (key == 'l') {
            learn_mode = !learn_mode;
            if (learn_mode) {
                cout << "Learn mode" << endl;
            } else {
                cout << "Work mode" << endl;
            }
        }
        // Press 'Up' and 'Down' to alter the color.
        if (learn_mode) {
            if (key == 63232) {
                learn_color++;
                if (learn_color == COLOR_SIZE)
                    learn_color = 0;
                cout << "learn color: " << ColorName[learn_color] << endl;
            }
            if (key == 63233) {
                learn_color--;
                if (learn_color == -1)
                    learn_color = COLOR_SIZE - 1;
                cout << "learn color: " << ColorName[learn_color] << endl;
            }
        }
        // Press ‘s’ to switch to snap shot mode.
        if (!snap_mode && key == 's') {
            snap_mode = true;
            cout << "Snap mode" << endl;
        }
        
        // Grab the color frame.
        freenect_sync_get_video(&data, &timestamp, 0, FREENECT_VIDEO_RGB);
        // Pack the data to matrix.
        Mat mat_color(480, 640, CV_8UC3, data);
        // Smooth the image.
        blur(mat_color, mat_color, Size(3, 3));
        // If using small screen, shrink the matrix.
        if (QVGA) {
            resize(mat_color, mat_color, Size(320, 240));
        }
        // Convert to BGR format.
        cvtColor(mat_color, mat_color, CV_RGB2BGR);
        
        // Grab the depth frame.
        freenect_sync_get_depth(&data, &timestamp, 0, FREENECT_DEPTH_REGISTERED);
        // Pack the data to matrix.
        Mat mat_depth(480, 640, CV_16UC1, data);
        // If using small screen, shrink the matrix.
        if (QVGA) {
            resize(mat_depth, mat_depth, Size(320, 240));
        }
        
        // Define a set of joints.
        vector<Joint> joints(JOINT_SIZE);
        
        if (!learn_mode) {
            // Calculate the positions of all joints.
            mask_color_by_depth(mat_color, mat_depth, mat_background, joints);
        } else {
            // Color learning.
            int hand_x = 0;
            int hand_y = 0;
            mask_color_by_depth(mat_color, mat_depth, mat_background, joints,
                                (ColorType)learn_color, &hand_x, &hand_y);
            // If detected the color.
            if (hand_x != 0 && hand_y != 0) {
                uchar* ptr_hand_color = mat_color.ptr(hand_y) + 3 * hand_x;
                Vec3 one_color;
                bgr_to_hsv(ptr_hand_color[0], ptr_hand_color[1], ptr_hand_color[2],
                           one_color.x, one_color.y, one_color.z);
                color_list.push_back(one_color);
                // Anylasis data of about one minute(66.7 seconds).
                if (color_list.size() == 2000) {
                    // Do color learning.
                    machine_learning(color_list,
                                     0.5f,
                                     0.05f,
                                     (ColorType)learn_color,
                                     "hsv_learn.txt");
                    // Prepare for the next color learning.
                    color_list.clear();
                }
            }
        }
        
        if (snap_mode) {
            // Time elapsed.
            snap_counter--;
            cout << snap_counter << endl;
            // Time out.
            if (snap_counter <= 0) {
                // Save the screen shot.
                imwrite("snap_shot.png", mat_color);
                // Reset snap counter.
                snap_counter = 150;
                snap_mode = false;
                cout << "Work mode" << endl;
            }
        }
        
        // Add the positions of all joints.
        m_pKinectBVH->AddAllJointsPosition(&joints[0]);
        
        // Increase the frame number.
        m_pKinectBVH->IncrementNbFrames();
        
        // Show the image.
        imshow("rgb", mat_color);
    }
    
    // Generate filename with current time.
    time_t nowtime = time(NULL);
    struct tm* local = localtime(&nowtime);
    char buf[256];
    sprintf(buf, "%d-%d-%d-%d-%d-%d.bvh",
            local->tm_year + 1900, local->tm_mon + 1, local->tm_mday,
            local->tm_hour, local->tm_min, local->tm_sec);
    
    // Save the motion capture data to bvh file.
    m_pKinectBVH->SaveToBVHFile(buf);
    
    // Clean the motion capture.
    delete m_pKinectBVH;
    m_pKinectBVH = NULL;
    
    // Clean the background buffer.
    delete[] background_buffer;
    background_buffer = NULL;
    
    // Close the kinect device.
    freenect_sync_stop();
    
    return 0;
}
