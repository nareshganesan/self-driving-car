#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

pwd = '/home/ng/udacity/nanodegree/self-driving-car/CarND-LaneLines-P1'

#reading in an image
image = mpimg.imread(pwd + '/test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    


    # draw lane lines
    for line in (left_lane, right_lane):
        line.draw(img)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def filter_lines(lines, image_shape):
    lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
    filtered_lines = []
    for line in lines:
        # only slope between 30 to 60 degrees
        if 0.5 <= np.abs(line.slope) <= 2:
            filtered_lines.append(line)
            
    pos_slope_lines = [line for line in filtered_lines if line.slope > 0]
    neg_slope_lines = [line for line in filtered_lines if line.slope < 0]
    
    # np.median used for lane approximation
    left_bias = np.median([line.bias for line in neg_slope_lines]).astype(int)
    left_slope = np.median([line.slope for line in neg_slope_lines])
    x1, y1 = 0, left_bias
    x2, y2 = -np.int32(np.round(left_bias / left_slope)), 0
    left_lane = Line(x1, y1, x2, y2)
    
    
    right_bias = np.median([line.bias for line in pos_slope_lines]).astype(int)
    right_slope = np.median([line.slope for line in pos_slope_lines])
    x1, y1 = 0, right_bias
    x2, y2 = np.int32(np.round((image_shape[0] - right_bias) / right_slope)), image_shape[0]
    right_lane = Line(x1, y1, x2, y2)
    return left_lane, right_lane

def lanes_image(lanes, image_shape):
    lanes_img = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for lane in lanes:
        lane.draw(lanes_img)
    return lanes_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# line object with helper slope 
class Line:
    """
    line connecting (x1, y1) and (x2, y2)
    # https://www.mathsisfun.com/algebra/line-equation-2points.html
    """
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self._slope()
        self.bias = self._bias()

    def _slope(self):
        # line slope https://www.mathsisfun.com/geometry/slope.html
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def _bias(self):
        # from slope equation => bias => y = mx + b => b = (y - mx)
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, img, color=[0, 255, 0], thickness=2):
        cv2.line(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)

import os
os.listdir(pwd + "/test_images/")

def read(src):
    return mpimg.imread(src)

def pipeline(image):
    # step 1: get grayscale image
    gray = grayscale(image)

    # step 2: gaussian smoothing of gray scale image
    gaussian_kernel = 5
    gaussian_blur_image = gaussian_blur(gray, gaussian_kernel)

    # step 3: canny transform
    low = 50
    high = 150
    canny_tranformed_image = canny(gaussian_blur_image, low, high)

    # step 4: hough transform to detect edges on masked image
    rho, theta, threshold, min_line_len, max_line_gap = 1, np.pi/180, 65, 40, 230
    lines = hough_lines(canny_tranformed_image, rho, theta, threshold, min_line_len, max_line_gap)

    # step 5: filter lines which have slope between 30 to 60 and only use median
    filtered_lines = filter_lines(lines, image.shape)

    # step 6: create lanes image
    lanes_img = lanes_image(filtered_lines, image.shape)

    # step 7: region of interest
    # mask region
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    # region vertices
    left_bottom = (0, image_height)
    left_top = (int(image_width/2 - image_width*.01), image_height/2 + int(image_height*.1))
    right_top = (int(image_width/2 + image_width*.01), image_height/2 + int(image_height*.1))
    right_bottom = (image_width, image_height)
    
    # mask region boundary vertices
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    masked_region = region_of_interest(lanes_img, vertices)
    
    return weighted_img(masked_region, image)

import glob
import os

image_dir = pwd + '/test_images'
count = len(glob.glob(os.path.join(image_dir, "*")))
idx = 0
rows = count / 3
cols = 3
# plt.figure(figsize=(20,10))
# for file in glob.glob(os.path.join(image_dir, "*")):
#     plt.subplot(rows,cols,idx+1)    # the number of images in the grid is 5*5 (25)
#     lanes = pipeline(read(file))
#     plt.imshow(lanes)
#     idx += 1

# plt.show()

white_input = pwd + '/test_videos/solidWhiteRight.mp4'
status = True
cap = cv2.VideoCapture(white_input)
frames = []
shape = None
while(status):
    # Capture frame-by-frame
    status, frame = cap.read()
    if not shape and frame is not None:
        shape = frame.shape
    frames.append(frame)
cap.release()

shape = [540, 960]
white_output = pwd + "/test_videos_output/solidWhiteRight.mp4"
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(white_output,fourcc, 20.0, (shape[0],shape[1]))
for frame in frames:
    out.write(frame)
    cv2.imshow('Frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('s'): 
        break
out.release()
cv2.destroyAllWindows() 

# plt.show()

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    return pipeline(image)

# white_output = pwd + '/test_videos_output/solidWhiteRight.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip(pwd + "/test_videos/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)


# yellow_output = pwd + '/test_videos_output/solidYellowLeft.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
# clip2 = VideoFileClip(pwd + '/test_videos_output/solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)

# challenge_output = pwd + '/test_videos_output/challenge.mp4'
# challenge_output = pwd + 'lane_detection_test_video_youtube.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
# # clip3 = VideoFileClip('test_videos/challenge.mp4')
# clip3 = VideoFileClip(pwd + '/lane_detection_test_video_youtube.mp4')
# challenge_clip = clip3.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)
