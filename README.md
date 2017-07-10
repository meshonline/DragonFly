# DragonFly
Kinect motion capture (Mocap) research project by color tracking.

![](mocap.png)

I'm studing how to use Kinect xbox 360 to achieve a low cost motion capture system.

I use both color and depth information to locate the space positions of all joints.

The algorithm does not use any statistic idea, so the positions of all joints are accurate.

I use wrist bands (13 colors) to mark colors for joints, they had only costed me five dollars.

![](wrist-bands.png)

I recommend that you put the light source near the camera, and let it lighten your body from the camera's view.

You must take a photo on these wrist bands, than write down their hues, satuations and values, you may use gimp to get the information.

Then you must modify 'colortrack.h', fill your colors into the color table manually.

To fine tune the colors to fit your light source, you can use the built-in color learning function to fine tune the colors.

The project is not mature at present, play it around at your own risk.

### Features:
1. Locate positions of all joints directly by color tracking.
2. Global optimized solution for color tracking.
3. Mixture gaussian model for color learning.
4. Freely turn around your body in 360 degrees.
5. Export motion capture data to 'bvh' format automatically.

### Support systems:
1. macOS
2. Linux
3. Windows.

### Usage:
1. Install the latest Freenect driver.
2. Install OpenCV 2.x
3. Set path for include directory and lib directory.
4. Mark joints with different colors.
5. Plug the Kinect xbox 360 device into the USB port.
6. Build and run.
7. [Optional]Fine tune the colors.
   Press 'l' key to switch the mode, 'Up' and 'Down' arrow key to change the color.
8. Perform actions.
9. Press 'esc' key to exit.

### To do:
1. [Solved]The hip center and the neck are difficult to mark colors, consider to generate them automatically.
2. The result is terrible due to the poor light source.
