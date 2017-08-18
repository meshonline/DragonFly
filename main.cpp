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
    
    bool snap_mode = false;
    int snap_counter = 300;
    unsigned long counter = 0l;
    Color_Tracker color_tracker;
    ofstream fout(".temp_star_point");
    while (true) {
        // Grab the color frame.
        freenect_sync_get_video(&data, &timestamp, 0, FREENECT_VIDEO_RGB);
        // Pack the data to matrix.
        Mat mat_color(480, 640, CV_8UC3, data);
        // Smooth the image.
        blur(mat_color, mat_color, Size(3, 3));
        // Convert to BGR format.
        cvtColor(mat_color, mat_color, CV_RGB2BGR);
        
        // Grab the depth frame.
        freenect_sync_get_depth(&data, &timestamp, 0, FREENECT_DEPTH_REGISTERED);
        // Pack the data to matrix.
        Mat mat_depth(480, 640, CV_16UC1, data);
        
        // Work mode
        if (!snap_mode) {
            // Generate star points for all joints.
            color_tracker.generate_star_points(mat_color, mat_depth, mat_background, fout);
            counter++;
        }
        // Snap mode
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
        
        // Show the image.
        imshow("rgb", mat_color);
        
        int key = waitKey(1);
        
        // Press 'ESC' to quit.
        if (key == 27) {
            break;
        }
        // Press ‘s’ to switch to snap shot mode.
        if (key == 's') {
            snap_mode = !snap_mode;
            if (snap_mode) {
                cout << "Snap mode" << endl;
            } else {
                cout << "Work mode" << endl;
            }
        }
    }
    fout.close();
    
    if (!snap_mode) {
        // Create a motion capture.
        KinectBVH kinect_bvh;
        
        // Generate T pose skeleton.
        kinect_bvh.CalibrateSkeleton();
        
        ifstream fin(".temp_star_point");
        while (counter > 0) {
            // Define a set of joints.
            vector<Joint> joints(JOINT_SIZE);
            
            // Calculate the positions of all joints.
            color_tracker.process_star_points(joints, fin);
            counter--;
            if (counter % 30 == 0) {
                cout << counter/30 << endl;
            }
            
            // Add the positions of all joints.
            kinect_bvh.AddAllJointsPosition(&joints[0]);
            
            // Increase the frame number.
            kinect_bvh.IncrementNbFrames();
        }
        fin.close();
        
        // Generate filename with current time.
        time_t nowtime = time(NULL);
        struct tm* local = localtime(&nowtime);
        char buf[256];
        sprintf(buf, "%d-%d-%d-%d-%d-%d.bvh",
                local->tm_year + 1900, local->tm_mon + 1, local->tm_mday,
                local->tm_hour, local->tm_min, local->tm_sec);
        
        // Save the motion capture data to bvh file.
        kinect_bvh.SaveToBVHFile(buf);
    }
    
    unlink(".temp_star_point");
    
    // Clean the background buffer.
    delete[] background_buffer;
    background_buffer = NULL;
    
    // Close the kinect device.
    freenect_sync_stop();
    
    return 0;
}
