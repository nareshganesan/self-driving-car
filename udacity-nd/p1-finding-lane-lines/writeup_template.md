# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

- Make a pipeline that finds lane lines on the road
- Reflect on your work in a written report

[//]: # "Image References"
[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

```
Algorithm:
    1. Convert the image to gray scale
    2. blurring the images to get better edges - Gaussian blur
    3. Apply canny edge detector to get edges detected
    4. Create a mask image with lanes - narrowing down the lane lines only to the region of interest
    5. Apply hough transformation to get lines
    6. Filter lines
        i. identify left and right lanes using simple linear algebra
            using mid point of image and min & max allowed slope within left and right side of the point of view.
    7. create left and right lanes using numpy polyfit function
    8.  i. find intersection point of right and left lanes using numpy linear algebra function
    9.  ii. add the line segment to buffer
    10. draw the lines on the images
```

![alt text][image1]

### 2. Identify potential shortcomings with your current pipeline

The current pipeline has problems in case of bad lighting and curved roads. Also the intersection points and lanes lines are misaligned drastically during begining of the video.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
