from helper import *


class Pipeline:
    def __init__(self, cal_path, nx, ny, debug=False):
        self.debug = debug
        self.mtx, self.dist = camera_calibration(cal_path, nx, ny, debug)        
        self.lines_fit = None
        self.left_points = []
        self.right_points = []

    def __call__(self, image):
        mask = thresholded_mask(image, self.mtx, self.dist, self.debug)
        if self.debug == True:
            plot_result([image, mask], ['image', 'mask'], rows=1, cols=2)

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
        # warp image
        warped, M, Minv = warp(mask, src, dest)
        src_roi = mark_region_of_interest(image, src)
        warped_roi = mark_region_of_interest(warped, dest)
        if self.debug == True:
            plot_result([src_roi, warped_roi], ['image', 'warped'], rows=1, cols=2)

        histogram, leftx, rightx = find_histogram_peaks(warped)
        if self.debug == True:
            plt.plot(histogram)
            plt.show()
        
        if self.lines_fit == None:
            # print("fit polynomial",type(self.left_points), type(self.right_points))
            self.lines_fit, self.left_points, self.right_points, lanes_mask = fit_polynomial(image, warped)
        else:
            # print("search around poly",type(self.left_points), type(self.right_points))
            self.lines_fit, self.left_points, self.right_points, lanes_mask = search_around_poly(image, warped, self.lines_fit, self.left_points, self.right_points)

        curvature_rads = curvature_radius(leftx=self.left_points[0], rightx=self.right_points[0], img_shape = image.shape)
        offsetx = car_offset(leftx=self.left_points[0], rightx=self.right_points[0], img_shape=image.shape)
        lane_image = draw_lane(image, warped, self.left_points, self.right_points, Minv)
        if self.debug == True:
            plot_result([image, lane_image], ['image', 'lane'], rows=1, cols=2)

        out_img = add_metrics(lane_image, leftx=self.left_points[0], rightx=self.right_points[0])
        return out_img
