# DragonFly
Kinect motion capture (Mocap) research project by color tracking.

![](mocap.png)

I'm studing how to use Kinect xbox 360 to achieve a low cost motion capture system.

I use both color and depth information to locate the space positions of all joints.

The algorithm does not use any statistic idea, so the positions of all joints are accurate.

I use wrist bands (13 colors) to mark colors for joints, they had only costed me five dollars.

![](wrist-bands.png)

I recommend that you put the light source near the camera, and let it lighten your body from the camera's view.

You must take a photo on these wrist bands, than write down their hues, satuations and values, gimp or other image editors can help you, then you must hard code the hues, satuations, values into the basic color table.

To fine tune the colors to fit your own light source, you can use the built-in color learning function to tune the colors.

The project is not mature at present, play it around at your own risk.

### Features:
1. Locate positions of all joints directly by color tracking.
2. Global optimized solution for color tracking.
3. Mixture gaussian model for color learning.
4. Freely turn around your body in 360 degrees.
5. Export motion capture data to 'bvh' format.

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
The hip center and the neck are difficult to mark colors.
