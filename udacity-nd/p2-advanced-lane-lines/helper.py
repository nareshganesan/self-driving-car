import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def plot_result(imgs, names, rows=0, cols=0):
    '''
    matplotlib plot helper function to different size plots
    Args:
        images - list of images to be plot
        names - list of names of the plot
        rows - no of rows of plot
        cols - no of cols of plot
    Returns:
        None
    '''
    if (len(imgs)==0 or len(names)==0): 
        return -1
    f, ax = plt.subplots(rows, cols, figsize=(16,8))
    f.tight_layout()
    i = 0
    if rows <= 1 :
        for c in range(cols):
            ax[c].imshow(imgs[i], cmap='gray')
            ax[c].set_title('{}'.format(names[i]), fontsize=24)
            ax[c].axis('off')
            i += 1
    else:
        for r in range(rows):
            for c in range(cols):
                ax[r, c].imshow(imgs[i])
                ax[r, c].set_title('{}'.format(names[i]), fontsize=24)
                ax[r, c].axis('off')
                i += 1
    plt.show()

def get_frames_from_video(video_path, images_path=None, skip=5):
    count = 0
    images = []
    names = []
    reader = cv2.VideoCapture(video_path)
    if reader.isOpened() == False:
        print("Error opening video stream or file")
        return images

    while reader.isOpened():
        ret, frame = reader.read()
        if ret == True:
            images.append(frame)
            name = "frame%d.jpg" % count
            names.append(name)
            if images_path and count%skip==0:
                cv2.imwrite(images_path + name, frame) 
            count = count + 1
        else:
            break
    reader.release()
    return images, names

def create_video_from_frames(images, filename, codec="MJPG"):
    try:
        height, width, channels = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(filename,fourcc, 10, (width,height))

        for image in images:
            writer.write(image)
        writer.release()
        return True
    except Exception as e:
        print(e)
        return False

def write_images_to_folder(images, image_names, file_path):
    '''
    writes the list of RGB / Grayscale / Binary images to file
    Args:
        images - list of RGB channel images
        image_names - list of names for images
        file_path - path to write the images
    Returns:
        status - bool status about write status
    '''
    count = 0
    for name, image in zip(image_names, images):
        if len(image.shape) > 2:
            # RGB image
            cv2.imwrite(file_path + name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        elif np.unique(image).tolist() == [0, 1]:
            # binary image
            cv2.imwrite(file_path + name, image * 255)
        else:
            # grayscale image
            cv2.imwrite(file_path + name, image)
        count += 1
    return len(images) == count

def resize(image, height=300, width=400, debug=False):
    '''
    resizes images
    Args:
        image - opencv RGB image
        height - height of resized image
        width - width of resized image
        debug - bool, displays the resized image
    Returns:
        resized opencv image
    '''
    if len(image.shape) > 2:
        # height, width, channels
        (h, w, d) = image.shape
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (width, height))
    else:
        # grayscale
        resized = cv2.resize(image, (width, height))
    if debug == True:
        cv2.imshow('resize', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return resized

def channels(image, debug=False):
    '''
    displays RGB, HLS channels individual channel images
    Args:
        image - opencv RGB image
        debug - bool, displays the resized image
    Returns:
        None
    '''
    image = resize(image)
    # RGB
    rgb_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    rgb = np.hstack((r,g,b))

    # HLS
    hls_gray = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h = hls_gray[:, :, 0]
    l = hls_gray[:, :, 1]
    s = hls_gray[:, :, 2]
    hls = np.hstack((h,l,s))

    all_channels = np.vstack((rgb, hls))

    if debug == True:
        cv2.imshow('all_channels', all_channels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def camera_calibration(images_path=None, nx=8, ny=6, debug=False, output_path="./output_images/"):
    '''
    Given calibration chessboard images, Returns camera matrix and distortion coefficients
    Args:
        image_path - path to calibration images
        nx - no of rows in chessboards to detect
        ny - no of cols in chessboards to detect
        debug - bool displays the images
    Returns:
        camera matrix
        distortion coefficients
    '''
    # list all calibration images
    calibration_images = glob.glob(images_path)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y co rodinates
    # image and object points
    images, names, img_shape, imgpoints, objpoints, fr, fc = [], [], None, [], [], 0, 0
    for idx, img in enumerate(calibration_images):
        filepath, filename = os.path.split(img)
        image = cv2.imread(img)
        if img_shape == None:
            img_shape = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            images.append(image)
            names.append(filename)
    rows = len(images) // 4
    cols = 4
    if debug == True:
        # plot_result(images, names, rows=rows, cols=cols)
        count = 1
        for name, img in zip(names, images):
            write_images_to_folder([img], [f"calibration_output{count}.jpg"], output_path)
            cv2.imshow('calibrationTesting', img)
            cv2.waitKey(0)
            count += 1
        cv2.destroyAllWindows()
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return mtx, dist

def undistort(image, mtx, dist, debug=False):
    '''
    Given image, camera matrix and distortion coefficients Returns undistorted image
    Args:
        image - 3 channel distorted image 
        mtx - camera matrix
        dist - distortion coefficient
        debug - bool displays the images
    Returns:
        undistorted opencv image
    '''
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    if debug == True:
        vis = np.concatenate((image, dst), axis=1)
        cv2.imshow('undistort', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return dst

def gaussian_blur(image, kernel=5, debug=False):
    '''
    applies gaussian blur to given image
    Args:
        image - 3 channel image
    Returns:
        image - 3 channel blurred image
    '''
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    if debug == True:
        rx_image = resize(image)
        rx_blurred = resize(blurred)
        vis = np.concatenate((rx_image, rx_blurred), axis=1)
        cv2.imshow('orig , gaussian blur', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return blurred

def sobel(image, gray, sx=False, sy=False, kernel=5, thresh_min=0, thresh_max=255, debug=False):
    '''
    sobel opertor on images. options
    1. sobel x
    2. sobel y
    3. sobel direction
    Args:
        img - 3 channel
        orient - sobel derivative direction x/y
        kernel - sobel kernel size
        thresh_min - minimum threshold for sobel 
        thresh_max - maximum trheshold for sobel 
    Returns:
        binary image
    '''
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel))
    if sx:
        sobel = np.uint8(255*sobelx/np.max(sobelx))
    elif sy:
        sobel = np.uint8(255*sobely/np.max(sobely))
    else:
        direction = np.arctan2(sobely, sobelx)
        sobel = np.absolute(direction)
    binary = np.zeros_like(sobel)
    binary[(sobel >= thresh_min) & (sobel <= thresh_max)] = 1
    if debug == True:
        rx_sobel = resize(sobel)
        rx_binary = resize(binary)
        plot_result([image, rx_sobel, rx_binary], ['image', 'channel', 'binary'], rows=1, cols=3)
    return binary

def thresholded_mask(image, mtx, dist, debug=False, output_path="./output_images/"):
    '''
    thresholded binary image using sobel x on rgb, hls color space and sobel direction
    Args:
        image - 3 channel RGB image
        mtx - camera matrix
        dist - distortion coefficients
        debug - debug flag
    Returns:
        image - binary image
    '''
    # undistory image
    undistorted = undistort(image, mtx, dist, debug)
    if debug == True:
        write_images_to_folder([undistorted], [f"undistorted.jpg"], output_path)
    # apply gaussian blur
    blurred = gaussian_blur(undistorted, kernel=3, debug=debug)
    # sobel x for rgb gray scale image
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    sx_gray_rgb = sobel(blurred, gray, sx=True, sy=False, kernel=5, thresh_min=20, thresh_max=100, debug=debug)
    # sobel x for l & s channel from hls color space
    hls = cv2.cvtColor(blurred, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    s = hls[:,:,2]
    sx_l_hls = sobel(blurred, l, sx=True, sy=False, kernel=5, thresh_min=20, thresh_max=100, debug=debug)
    sx_s_hls = sobel(blurred, s, sx=True, sy=False, kernel=5, thresh_min=20, thresh_max=100, debug=debug)
    # combining all the sobel x from different feature images
    sobel_color = sx_gray_rgb | sx_l_hls | sx_s_hls

    sobel_direction = sobel(blurred, gray, sx=False, sy=False, kernel=3, thresh_min=np.pi / 6, thresh_max=np.pi / 2, debug=debug)
    sobel_combined = (sobel_color == 1) & (sobel_direction == 1)
    sobel_combined = np.float32(sobel_combined)

    # if debug == True:
    #     plot_result([image, sx_gray_rgb, sx_l_hls, sx_s_hls, sobel_combined], ['image', 'sx_gray', 'sx_l', 'sx_s', 'sobel_combined'], rows=1, cols=5)
    
    mask = np.float32(np.zeros_like(sobel_combined))
    # region of interest
    vertices = np.array(
        [[0, height - 1], [width / 2, int(0.5 * height)], [width - 1, height - 1]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [vertices], 1)
    thresholded = cv2.bitwise_and(sobel_combined, mask)

    if debug == True:
        plot_result([image, sx_gray_rgb, sx_l_hls, sx_s_hls, sobel_combined, thresholded], ['image', 'sx_gray', 'sx_l', 'sx_s', 'sobel_combined', 'thresholded'], rows=1, cols=6)
    return thresholded

def warp(image, src, dest):
    # image size
    image_size = (image.shape[1], image.shape[0])
    # perpective transform
    M = cv2.getPerspectiveTransform(src, dest)
    # inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(
        image, M, image_size, flags=cv2.INTER_LINEAR
    )  # INTER_NEAREST - keep same size as input image
    return warped, M, Minv

def mark_region_of_interest(image, src):
    '''
    marks region of interest given image and src coordinates
    Args:
        image - 3 channel image RGB
        src - four vertices of region
    Returns:
        image - 3 channel image RGB
    '''
    copy = np.copy(image)
    color = [255, 0, 0]  # Red
    # points
    thickness = -1
    radius = 10
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.circle(copy, (x0, y0), radius, color, thickness)
    cv2.circle(copy, (x1, y1), radius, color, thickness)
    cv2.circle(copy, (x2, y2), radius, color, thickness)
    cv2.circle(copy, (x3, y3), radius, color, thickness)
    # lines
    thickness = 2
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.line(copy, (x0, y0), (x1, y1), color, thickness)
    cv2.line(copy, (x1, y1), (x2, y2), color, thickness)
    cv2.line(copy, (x2, y2), (x3, y3), color, thickness)
    cv2.line(copy, (x3, y3), (x0, y0), color, thickness)
    return copy

def find_histogram_peaks(image):
    '''
    Find histogram peaks corresponding to image features
    Args:
        image - binary image
    Returns:
        histogram - features
        leftx - left lane base point
        rightx - right lane base point
    '''
    histogram = np.sum(image[image.shape[0] // 2 :, :], axis=0)
    half_width = np.int(histogram.shape[0] // 2)
    # left lane
    leftx = np.argmax(histogram[:half_width])
    # right lane
    rightx = np.argmax(histogram[half_width:]) + half_width
    return histogram, leftx, rightx

def find_lane_pixels(image, binary_warped, debug=False):
    '''
    find the left and right lane pixels using a sliding window approach. 
    1. Split the ROI into fixed number of windows (hyperparameter)
    2. Find boundaries of window using histogram peaks. ( )
        1. leftx-margin
        2. leftx+margin
        3. lefty_low = image_height - window_no+1 * (image_height//nwindows)
        4. lefty_high = image_height - window_no * (image_height//nwindows)
    3. find all the bright pixels within this sliding window
    4. find the mean of all the pixels within the window
    5. find the leftx, lefty, rightx, righty
    Args:
        image - 3 channel source image
        binary_warped - binary image
        debug - optional debug bool flag
    Returns:
        leftx - left lane base x coordinate
        lefty - left lane base y coordinate
        rightx - righ lane base x coordinate
        righty - right lane base y coordinate
        lane_pixels - 3 channel mask of the lane lines
    '''
    # Create an output image to draw on and visualize the result
    lane_pixels = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    histogram, leftx_base, rightx_base = find_histogram_peaks(binary_warped)

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(lane_pixels,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(lane_pixels,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if debug == True:
        rximage = resize(image)
        rxlanepixels = resize(lane_pixels)
        vis = np.concatenate((rximage, rxlanepixels), axis=1)
        # cv2.imshow('lanes pixels', vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plot_result([rximage, rxlanepixels], ['image', 'lanes pixels'], rows=1, cols=2)
    return leftx, lefty, rightx, righty, lane_pixels

def fit_polynomial(image, binary_warped, debug=False):
    '''
    fit a polynomial line using lane pixels coordinates found using sliding window
    Args:
        image - source image
        binary_warped - binary mask image
        debug - optional bool debug flag
    Returns:
        TODO
    '''
    # Find our lane pixels first
    leftx, lefty, rightx, righty, lane_pixels = find_lane_pixels(image, binary_warped, debug)

    lane_mask = np.copy(lane_pixels)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    lane_mask[lefty, leftx] = [255, 0, 0]
    lane_mask[righty, rightx] = [0, 0, 255]

    for index in range(binary_warped.shape[0]):
        cv2.circle(lane_mask, (int(left_fitx[index]), int(ploty[index])), 3, (255,255,0))
        cv2.circle(lane_mask, (int(right_fitx[index]), int(ploty[index])), 3, (255,255,0))
    if debug == True:
        rximage = resize(image)
        rxlanepixels = resize(lane_pixels)
        rxlanemask = resize(lane_mask)
        vis = np.concatenate((rximage, rxlanepixels, rxlanemask), axis=1)
        cv2.imshow('sliding window lanes', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [left_fit, right_fit], [left_fitx, ploty], [right_fitx, ploty], lane_mask

def get_averaged_line(previous_lines, new_line):
    '''
    moving average of previous frame lines
    '''
    # moving average window size
    num_frames = 12
    if new_line is None:
        # no previous lines
        if len(previous_lines) == 0:
            return previous_lines, None
        else:
            # return last found previous line
            return previous_lines, previous_lines[-1]
    else:
        # buffer lines less then moving average window size
        if len(previous_lines) < num_frames:
            previous_lines.append(new_line)
            return previous_lines, new_line
        else:
            # buffer lines greater than moving average window size
            # maintain moving average window
            previous_lines[0:num_frames-1] = previous_lines[1:]
            previous_lines[num_frames-1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += previous_lines[i]
            new_line /= num_frames
            return [previous_lines, new_line]

def search_around_poly(image, binary_warped, lines_fit=None, left_lines=[], right_lines=[], moving_avg=0, debug=False):
    '''
    find left and right lane pixels around the previous frame's left and right lane coordinates
    '''
    if lines_fit == None:
        return fit_polynomial(image, binary_warped, debug)

    left_fit = lines_fit[0]
    right_fit = lines_fit[1]
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    if (leftx.size == 0 or rightx.size == 0):
        return fit_polynomial(image, binary_warped, debug)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    
    # If no pixels were found return None
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Smoothing
    mean_difference = np.mean(right_fitx - left_fitx)
        
    if moving_avg == 0:
        moving_avg = mean_difference
        
    if (mean_difference < 0.7*moving_avg or mean_difference > 1.3*moving_avg):
        if len(left_lines) == 0 and len(right_lines) == 0:
            return fit_polynomial(image, binary_warped, debug)
        else:
            left_fitx = left_lines[-1]
            right_fitx = right_lines[-1]
    else:
        left_lines, left_fitx = get_averaged_line(left_lines, left_fitx)
        right_lines, right_fitx = get_averaged_line(right_lines, right_fitx)
        mean_difference = np.mean(right_fitx - left_fitx)
        moving_avg = 0.9*moving_avg + 0.1*mean_difference

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    for index in range(binary_warped.shape[0]):
        cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 3, (255,255,0))
        cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 3, (255,255,0))
    
    ## End visualization steps ##
    if debug == True:
        rximage = resize(image)
        rxout_img = resize(out_img)
        vis = np.concatenate((rximage, rxout_img), axis=1)
        cv2.imshow('sliding window lanes', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [left_fit, right_fit], [left_fitx, ploty], [right_fitx, ploty], out_img

def curvature_radius(leftx, rightx, img_shape, xm_per_pix=3.7/800, ym_per_pix=25/720):
    '''
    computed the lane curvature radius
    lane width - 12 feet or 3.7 meters
    dashed lane lines  10 feet or 3 meters
    '''
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # second order polynomial to pixel positions
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # x and y coordinates to meters
    ym_per_pix = 25 / 720  # meters per pixel in y dimension    250      1050 => 800
    xm_per_pix = 3.7 / 800  # meters per pixel in x dimension ---|--------|---

    # maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in meters
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculation of R_curve (radius of curvature)
    left_curverad = (
        (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * left_fit_cr[0])
    right_curverad = (
        (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)

def car_offset(leftx, rightx, img_shape, xm_per_pix=3.7/800):
    mid_imgx = img_shape[1] // 2
    car_pos = (leftx[-1] + rightx[-1]) / 2
    offsetx = (mid_imgx - car_pos) * xm_per_pix
    return offsetx

def draw_lane(image, warped_img, left_points, right_points, Minv):
    '''
    draw the lines back onto original image using inverse perspective transform
    '''
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    left_fitx = left_points[0]
    right_fitx = right_points[0]
    ploty = left_points[1]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    inverse = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    return cv2.addWeighted(image, 1, inverse, 0.3, 0)

def add_metrics(img, leftx, rightx, xm_per_pix=3.7 / 800, ym_per_pix=25 / 720):
    # Calculate radius of curvature
    curvature_rads = curvature_radius(
        leftx=leftx,
        rightx=rightx,
        img_shape=img.shape,
        xm_per_pix=xm_per_pix,
        ym_per_pix=ym_per_pix,
    )
    # Calculate car offset
    offsetx = car_offset(leftx=leftx, rightx=rightx, img_shape=img.shape)

    # Display lane curvature
    out_img = img.copy()
    cv2.putText(
        out_img,
        "Left lane curvature: {:.2f} m".format(curvature_rads[0]),
        (60, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        out_img,
        "Right lane curvature: {:.2f} m".format(curvature_rads[1]),
        (60, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Display car center deviation
    cv2.putText(
        out_img,
        "Center deviation: {:.2f} m".format(offsetx),
        (60, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    return out_img
