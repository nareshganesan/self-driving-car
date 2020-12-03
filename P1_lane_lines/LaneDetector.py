import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LaneDetector:
    def __init__(self,
        buffer_size=10,
        kernel_size=3,
        canny_low=20,
        canny_high=60,
        hough_rho=2,
        hough_theta=(np.pi/180),
        hough_threshold=50,
        hough_min_line_len=15,
        hough_max_line_gap=10,
        min_line_slope=0.5,
        max_line_slope=0.9,
        line_color=[255,0,0]
    ):
        self.buffer_size = buffer_size
        self.kernel_size = kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.min_line_slope = min_line_slope
        self.max_line_slope = max_line_slope
        self.line_color = line_color
        self.left_buffer = []
        self.right_buffer = []

    def add_to_buffer(self, line, buffer):
        '''
        maintain a queue of line buffer of size , buffer_size
        '''
        buffer.append(line)
        return buffer[-self.buffer_size:]

    def grayscale(self, img):
        '''
        get a gray scale image
        '''
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow_dark = np.array([15, 127, 127], dtype=np.uint8)
        yellow_light = np.array([25, 255, 255], dtype=np.uint8)
        yellow_range = cv2.inRange(img, yellow_dark, yellow_light)

        white_dark = np.array([0, 0, 200], dtype=np.uint8)
        white_light = np.array([255, 20, 255], dtype=np.uint8)
        white_range = cv2.inRange(img, white_dark, white_light)

        return cv2.bitwise_and(img, img, mask=(yellow_range | white_range))

    def get_mask_vertices(self, img):
        '''
        get mask vertices relative to input image shape
        '''
        # mask region
        image_height = img.shape[0]
        image_width = img.shape[1]
        
        # region vertices
        left_bottom = (0, image_height)
        left_top = (int(image_width/2 - image_width*.01), image_height/2 + int(image_height*.1))
        right_top = (int(image_width/2 + image_width*.01), image_height/2 + int(image_height*.1))
        right_bottom = (image_width, image_height)
        
        # mask region boundary vertices
        return np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    def get_masked_image(self, img):
        '''
        create a mask image using gaussian blur and canny edge detector
        '''
        gaussian = cv2.GaussianBlur(img[:,:,2], (self.kernel_size, self.kernel_size), 0)
        canny = cv2.Canny(gaussian, self.canny_low, self.canny_high)
        vertices = self.get_mask_vertices(canny)
        mask = np.zeros_like(canny)
        cv2.fillPoly(mask, vertices, 255) # ignore_mask_color = 255
        return cv2.bitwise_and(canny, mask)


    def filter_points(self, lines, midx):
        '''
        separate left and right lane coordinates using midpoint of image (x) and min & max slope value
        '''
        left_points = dict(x=[], y=[])
        right_points = dict(x=[], y=[])
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx != 0: # divide by zero
                slope = float(y2 - y1) / float(x2 - x1)
                if x1 < midx and x2 < midx: # left line -ve slope
                    if -self.max_line_slope < slope < -self.min_line_slope:
                        left_points['x'] += [x1, x2]
                        left_points['y'] += [y1, y2]
                elif x1 > midx and x2 > midx: # right line - +ve slope
                    if self.max_line_slope > slope > self.min_line_slope:
                        right_points['x'] += [x1, x2]
                        right_points['y'] += [y1, y2]
        return left_points, right_points

    def get_lines_from_points(self, left_points, right_points):
        '''
        create a left and right line from numpy's polyfit function and add the candidate line to buffer
        '''
        if len(left_points['x']) > 1:
            left_line = np.polyfit(left_points['x'], left_points['y'], 1)
            self.left_buffer = self.add_to_buffer(left_line, self.left_buffer)
        if len(right_points['x']) > 1:
            right_line = np.polyfit(right_points['x'], right_points['y'], 1)
            self.right_buffer = self.add_to_buffer(right_line, self.right_buffer)
        return (np.mean(self.left_buffer, axis=0), np.mean(self.right_buffer, axis=0),)

    def get_intersection_point(self, left_line, right_line):
        '''
        get intersection of left and right lines (using slope & intercept values) using numpy's linear algebra function
        '''
        left_slope, left_intercept = left_line
        right_slope, right_intercept = right_line

        a = [[left_slope, -1],
             [right_slope, -1]]
        b = [-left_intercept, -right_intercept]
        x, y = np.linalg.solve(a, b)
        return int(x)

    def get_line_segment(self, x1, x2, line):
        '''
        use slope and y-intercept to get y1 & y2 using numpy's poly1d function
        '''
        fx = np.poly1d(line)
        y1 = int(fx(x1))
        y2 = int(fx(x2))
        return ((x1, y1), (x2, y2))

    def get_lane_lines(self, img):
        '''
        using hough lines, filter, separate the left and right lanes, then get the intersection of the lanes to get the
        final left and right lanes
        '''
        left_x = 0
        right_x = img.shape[1]
        line_segments = []
        line_segments = cv2.HoughLinesP(img, self.hough_rho, self.hough_theta, self.hough_threshold, self.hough_min_line_len, self.hough_max_line_gap)
        left_points, right_points = self.filter_points(line_segments, int(right_x / 2))
        left_line, right_line = self.get_lines_from_points(left_points, right_points)
        intersect_x = self.get_intersection_point(left_line, right_line)
        return (self.get_line_segment(left_x, intersect_x, left_line),
                self.get_line_segment(right_x, intersect_x, right_line),)

    def detect_lanes(self, img):
        '''
        detect lanes using the following pipeline
        1. grayscale
        2. mask the grayscale image
            a. gaussian blur
            b. canny edge detection
            c. selecting only a region of interest and masking other values
        3. detect left and right lanes using simple linear algebra concepts using numpy functions
        '''
        filtered_img = self.grayscale(img)
        masked = self.get_masked_image(filtered_img)
        lane_lines = self.get_lane_lines(masked)
        line_img = np.zeros_like(img)
        for line in lane_lines:
            cv2.line(line_img, line[0], line[1], self.line_color, 3)
        return cv2.addWeighted(img, 0.8, line_img, 1.0, 0)

# path = "test_images/"
# output_path = "test_images_output/"
# files = os.listdir(path)

# ld = LaneDetector()

# for file in files:
#     if file.endswith(".jpg") or file.endswith(".png"):
#         img = mpimg.imread(path + file)
#         result = ld.detect_lanes(img)
#         plt.imshow(result, cmap="gray")
#         plt.show()
#         cv2.imwrite(output_path + file, result)
