# DragonFly
Kinect motion capture (Mocap) research project by color tracking.

Use Kinect xbox 360 to achieve a low cost motion capture system.

### Features:
1. Locate joints directly by color tracking.
2. Global optimized solution for color tracking.
3. Mixture gaussian model for color tracking.
4. Turn body around in 360 degrees freely.
5. Export motion capture to 'BVH' format.

### Support systems:
1. macOS
2. Linux
3. Windows.

### Usage:
1. Plug the Kinect xbox 360 device into the USB port.
2. Install the latest Freenect driver.
3. Install OpenCV 2.x
4. Set path for include directory and lib directory.
5. Build and run.
6. Mark joints with different colors.
7. Perform actions.
8. Press 'ESC' to exit.

### To do:
1. The hip center and the neck are difficult to mark colors, generate their positions automatically.
2. Load basic color table from file other than hard coding.
