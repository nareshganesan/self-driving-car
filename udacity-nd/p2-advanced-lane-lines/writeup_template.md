## Advanced lane lines project

---

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/test1.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"
[//]: # "References"
[camera calibration]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L150 "Camera calibration"
[thresholded_mask]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L267 "thresholded_mask"
[lane_lines.py]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/lane_lines.py#L31 "warp"
[warp]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L315 "warp"
[find_lane_pixels]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L379 "find_lane_pixels"
[fit_polynomial]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L485 "fit_polynomial"
[curvature_radius]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L674
[car_offset]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L712
[draw_lane]: https://github.com/nareshganesan/self-driving-car/blob/main/udacity-nd/p2-advanced-lane-lines/helper.py#L718

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera matrix and distortion coefficients were calculated using the following opencv functions. we use real world object points and image points of the corners identified by the opencv function on the chessboard images to calibrate the camera.
[Camera calibration] is implemented in the `helper.py`

```bash
# identifies the corners from the given grayscale image of chessboard
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```

```bash
# given the object and images points from previous opencv function, the following function computes the camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the above camera matrix and distortion coefficients, I've attached the undistorted test image.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I've used the following channel images & sobel operator to create a thresholded binary image for identifying lanes.

`grayscale`, `l channel in HLS`, `s channel in HLS`

[thresholded_mask] method used to obtain the thresholded image can be found in `helper.py`

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After getting the thresholded image, I use the function [warp] in `helper.py` to get the perspective transform of the thresholded mask.
The warping function are called in `lane_lines.py` ([lane_lines.py])

```python
# region of interest coordinates
src = np.float32([
    [280,  700],  # Bottom left
    [595,  460],  # Top left
    [725,  460],  # Top right
    [1125, 700]   # Bottom right
])
dest = np.float32([
    [250,  720],  # Bottom left
    [250,    0],  # Top left
    [1065,   0],  # Top right
    [1065, 720]   # Bottom right
])
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 280, 700  |  250, 720   | # Bottom left |
| 595, 460  |   250, 0    | # Top left |
| 725, 460  |   1065, 0   | # Top right |
| 1125, 700 |  1065, 720  | # Bottom right |

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use the following functions [find_lane_pixels] and [fit_polynomial] to identify lane line pixels using histogram and sliding window approach.
Results of lane pixels are used to fit a polynomial line.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The following functions [curvature_radius] and [car_offset] in `helper.py` to compute the radius of curvature and car offset from center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I've attached an example image of the drawn lane image. [draw_lane] function in `helper.py` to draw the lane on source image.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

My solution works well on the project video but fails badly on the challenge and harder challenge videos. I believe the feature selection of the thresholded binary image is the one of the main source of failure. I will be playing more with colors, sobel operators and threshold to be robust on different road conditions.
