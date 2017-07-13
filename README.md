# DragonFly
Kinect motion capture (Mocap) research project by color tracking.

![](mocap.png)

I'm studing how to use Kinect xbox 360 to achieve a low cost motion capture system.

In this project, I use both color and depth information to locate the positions of joints.

The algorithm does not use statistic method, so the positions of joints are definite and accurate.

I use inexpensive wrist bands (13 colors) to mark colors for joints, they only costed me less than five dollars.

But I found that my clothes are white, so I use a piece of brown non-woven fabric to instead the white wrist band finally.

Non-woven fabrics have more colors than wrist bands, and they are cheaper than wrist bands, but they are lack of elastic.

![](wrist-bands.png)

I recommend that you put the light source near the camera, and let it lighten your body from camera's angle.

You need to take a photo on the wrist bands under the light source, than write down their hues, satuations and values, you may use GIMP to get these information.

Then you need to modify 'colortrack.h', find the function of 'mask_color_by_depth()', fill the colors into the color table instead of mine.

You can also fine tune the colors to fit the light source by the built-in color learning function.

This step is optional, if the light is coming from the camera's angle, it is not necessary to fine tune the colors.

If you'd like to fine tune the colors, remove all the contents in hsv_learn.txt, generate your data instead of mine.

The project is far less mature at present, play around at your own risk.

![](snap_shot.png)

### Features:
1. Locate positions of joints directly by color tracking.
2. Global optimal solution for color recognition.
3. Mixture gaussian model for color learning.
4. Turn around your body in the range of 360 degrees.
5. Export motion capture data to 'BVH' format.

### Support systems:
1. macOS
2. Linux
3. Windows.

### Usage:
1. Install latest [libfreenect driver](https://github.com/OpenKinect/libfreenect/).
2. Install OpenCV 2.4.12
3. Mark joints with different colors.
4. Plug the Kinect xbox 360 device into the USB port.
5. Set path for include directory and lib directory.
6. Fill in the color table via modifying the source code.
7. Build and run.
8. Perform actions.
9. Press 'Esc' key to exit.

### To do:
[Solved]The result is poor in dim light.
