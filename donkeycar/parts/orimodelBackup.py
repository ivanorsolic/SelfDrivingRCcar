
from donkeycar.parts.keras import KerasPilot

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import LSTM, Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
import tensorflow as tf
import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import json

class OriModel(KerasPilot):
    def __init__(self, num_outputs=2, input_shape=(720, 1280, 3), roi_crop=(0, 0), *args, **kwargs):
        super(OriModel, self).__init__(*args, **kwargs)
        self.model = oriModel(input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer="adam",
                loss='mse')

    def run(self, img_arr):
        
        fitx, fity, ploty = processImage(img_arr)
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # lane = np.concatenate([fitx, fity])
        lane = np.array([fitx, fity])
        lane = np.concatenate(lane, 0)
        lane = lane.reshape((1,) + lane.shape)
        steering, throttle = self.model.predict([img_arr, lane])
        return steering[0][0], throttle[0][0]

def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return (new_height, input_shape[1], input_shape[2])


def oriModel(input_shape):
    
    img_in = Input(shape=input_shape, name='img_in')
    lane_in = Input(shape=(360,), name="lane_in")

    
    # Dropout rate
    keep_prob = 0.5
    rate = 1 - keep_prob

    y = lane_in

    y = LSTM(128, return_sequences=True, name="LSTM_seq")(y)
    y = Dropout(.1)(y)
    y = LSTM(128, return_sequences=True, name="LSTM_fin")(y)
    y = Dropout(.1)(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(.1)(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(10, activation='relu')(y)

    x = img_in
    
    # Convolutional Layer 1
    x = Convolution1D(filters=12, kernel_size=5, strides=(2, 2), input_shape = input_shape, activation='relu', name="Conv2D_First_layer")(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 2
    x = Convolution2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu', name="Conv2D_Second_layer")(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 3
    x = Convolution2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu', name="Conv2D_Third_layer")(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 4
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name="Conv2D_Fourth_layer")(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 5
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name="Conv2D_Fifth_layer")(x)
    x = Dropout(rate)(x)

    # Flatten Layers
    x = Flatten()(x)
    # Fully Connected Layer 1
    x = Dense(100, activation='relu')(x)
    z = concatenate([x, y])

    # Fully Connected Layer 2
    z = Dense(100, activation='relu')(z)

    # Fully Connected Layer 3
    z = Dense(50, activation='relu')(z)
    z = Dense(25, activation='relu')(z)
    z = Dense(10, activation='relu')(z)
    z = Dense(5, activation='relu')(z)
    z = Dense(2, activation='relu')(z)
    #categorical output of the angle
    angle_out = Dense(1, activation='linear', name='angle_out')(z)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    #continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(z)

    model = Model(inputs=[img_in, lane_in], outputs=[angle_out, throttle_out])
    
    return model
    
def warpImage(img):
    img_size = (320, 180)
    src_coordinates = np.float32(
            [[0,  180],  # Bottom left
            [112.5, 87.5], # Top left
            [200, 87.5], # Top right
            [307.5, 180]]) # Bottom right
            

    dst_coordinates = np.float32(
                [[80,  180],  # Bottom left
                [80,    0.25],  # Top left
                [230,   0.25],  # Top right
                [230, 180]]) # Bottom right   


    M = cv2.getPerspectiveTransform(src_coordinates, dst_coordinates)

    # Compute the inverse perspective transfor also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst_coordinates, src_coordinates)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, Minv
    
def combineThresholds(img, s_thresh=(170, 255), sx_thresh=(100, 150)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    #return s_binary
    return combined_binary

def histogram(img):
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
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

    return leftx, lefty, rightx, righty, out_img

left_fit = []
right_fit = []

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    global left_fit, right_fit
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    if len(left_fit) == 0:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
    left_lane_inds = (
        (nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))
    )
    right_lane_inds = (
        (nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
      & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
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
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return left_fitx, right_fitx, ploty

def overlayImagesForNeuralNet(img1, img2):
    from PIL import Image
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    img1 = img1.convert("RGBA")
    pixels = img1.getdata()

    newPixels = []
    for item in pixels:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newPixels.append((255, 255, 255, 0))
        else:
            newPixels.append(item)

    img1.putdata(newPixels)
    img2.paste(img1, (0, 0), img1)
    return np.array(img2.convert("RGB"))

def processImage(img): 
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    undistorted = img
    warped, Minv = warpImage(undistorted)
    '''combinedThreshold = combineThresholds(warped)
    left_fitx, right_fitx, ploty = search_around_poly(combinedThreshold, left_fit, right_fit)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combinedThreshold).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (51, 153, 255))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    result = overlayImagesForNeuralNet(newwarp, undistorted)
    # result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    return result'''
    return warped

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)