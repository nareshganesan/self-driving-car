import numpy as np
import cv2
import glob


def get_chessboard_corners(images, nx=8, ny=6, isgray=True):
    '''
    takes list of chessboard images in different angles with default 8*6 corners.
    Input image can be 3 channel or gray scale image
    '''

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y co rodinates

    imgpoints, objpoints = [], []
    for idx, img in images:
        if not isgray:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(image, (nx, ny),  corners, ret)
    
    return objpoints, imgpoints


def get_undistorted_image(image, objpoints, imgpoints, isgray=True):
    '''
    takes objpoints and image points, image shape rows, cols to return camera calibration matrix, dist coefficient, rvec, tvec
    '''
    if not gray:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1:],None,None)
    else:
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1],None,None)
    return cv2.undistort(image, mtx, dist, None, mtx)
    
