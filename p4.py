
# coding: utf-8

# ### Advanced Lane Finding Project:
# ### Detect lane lines using computer vision techniques. The goals of this project are the following:
# 
#     1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#     2. Apply a distortion correction to raw images.
#     3. Use color transforms, gradients, etc., to create a thresholded binary image.
#     4. Apply a perspective transform to rectify binary image ("bird's-eye view").
#     5. Detect lane pixels and fit to find the lane boundary.
#     6. Determine the curvature of the lane and vehicle position with respect to center.
#     7. Warp the detected lane boundaries back onto the original image.
#     8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# In[75]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import os


# In[76]:

def cam_calibrate(loc):
    
    #Read image frame wise using glob
    images = glob.glob(loc)
    
    # Creare 3D object points with 9*6 as chessboard size
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    corner = (9, 6)

    for image in images:
        img = mpimg.imread(image)
        # convert image to gray scame
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, corner, None)
        
        # if corners are found (i.e., ret == True) then add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    
    # output of cv2.calibratecamera function: 
    # dist = distortion co-effients
    # mtx = camera matrix to transform 3D object points to 2D image points
    # rvecs, tvecs = rotational and translational vectors - these gives position of the camera in the world 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist


# In[77]:

def undistort(distorted, mtx, dist):
    # cv2.undistort: Returns undistored image     
    undistored = cv2.undistort(distorted, mtx, dist, None, mtx)
    return undistored


# In[78]:

def warp(img):
    # Check frame prespective to get a top-down view of the lane
    warped = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


# In[79]:

def color_gradient_threshold(img):
    # Color and Gradient

    # 1. 
    # Convert to HLS color space: 
    # Hue as the value that represents color independent of any change in brightness.
    # Lightness represent different ways to measure the relative lightness or darkness of a color.
    # Saturation is a measurement of colorfulness
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    hue = hls[:, :, 0]
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]
    
    # 2. 
    # Applying the Sobel operator to an image is a way of taking the derivative 
    # of the image in the x or y direction.
    # Sobel for x and y:
    # Taking the gradient in the x-direction emphasizes edges closer to vertical 
    # and in the y-direction, edges closer to horizontal.
    
    # Filter size
    kszie=9
    
    # 2.a
    # The derivative in the x-direction (the 1, 0 at the end denotes x-direction)
    sobelx = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=kszie) 
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx) 
    
    #2. b
    # The derivative in the x-direction (the 0, 1 at the end denotes x-direction)
    sobely = cv2.Sobel(lightness, cv2.CV_64F, 0, 1, ksize=kszie)
    abs_sobely = np.absolute(sobely)
    
    # Threshold x gradient
    # Convert the absolute value image to 8-bit
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    grad_x = np.zeros_like(scaled_sobel)
    x_thresh=(40, 100)
    # Create a mask of 1's where the scaled gradient magnitude 
    grad_x[(scaled_sobel >= x_thresh[0]) & (scaled_sobel <= x_thresh[1])] = 1
    
    # 3. 
    # Direction of the gradient - we're interested only in edges of a particular orientation
    # Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of −π/2 to π/2. 
    # An orientation of 0 implies a horizontal line and orientations of +/−π/2 imply vertical lines.
    absgraddir=np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_thresh=(0.2, 1.2)
    dir_binary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    
    # 4. 
    # Threshold s channel
    saturation_threshold=(170, 255)
    s_binary = np.zeros_like(saturation)
    s_binary[(saturation >= saturation_threshold[0]) & (saturation <= saturation_threshold[1])] = 1

    # 5. 
    # Threshold h channel
    hue_threshold=(20, 100)
    h_binary = np.zeros_like(hue)
    h_binary[(hue >= hue_threshold[0]) & (hue <= hue_threshold[1])] = 1

#     # 6. Mask yellow and white colors
#     # Lane colors
#     yellow_lane= cv2.inRange(img, (200,200,0), (255,255,150))
#     white_lane= cv2.inRange(img, (200, 200, 200), (255, 255, 255))
#     #yellow_and_white_img = np.divide(yellow_and_white_img, 255)

#     # Combine all thresholds: use various aspects of your gradient measurements 
#     # (x, y, color channels, direction) to isolate lane-line pixels.
#     combined_binary = np.zeros_like(grad_x)
#     combined_binary[(h_binary == 1) | (yellow_lane | white_lane == 1) | ((grad_x == 1) & (dir_binary == 1)) ] = 1

    # Lane colors
    yellow_lane= cv2.inRange(img, (200,200,0), (255,255,150))
    white_lane= cv2.inRange(img, (200, 200, 200), (255, 255, 255))
    yellow_and_white_img = yellow_lane | white_lane
    yellow_and_white_img = np.divide(yellow_and_white_img, 255)

    # Combine all thresholds: use various aspects of your gradient measurements 
    # (x, y, color channels, direction) to isolate lane-line pixels.
    combined_binary = np.zeros_like(grad_x)
    combined_binary[(h_binary == 1) | (yellow_and_white_img == 1) | ((grad_x == 1) & (dir_binary == 1)) ] = 1

    
    return combined_binary

# Plotting thresholded images
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.set_title('Stacked thresholds')
# ax1.imshow(combined_binary)


# In[80]:

# Locate the Lane Lines and Fit a Polynomial

def detect_first_lane(binary_warped):
    
    # Remove the center of the image, where we are sure there are no lanes
    y = binary_warped.shape[0]
    x = binary_warped.shape[1]
#     center = int(x / 2)
#     offset = 200
    
#     a3 = np.array( [[[center-offset,0],[center+offset,0],[center+offset,y],[center-offset,y]]], dtype=np.int32 )
#     cv2.fillPoly( binary_warped, a3, 0 )

    
    # In thresholded binary image, pixels are either 0 or 1, 
    # so the two most prominent peaks in this histogram 
    # will be good indicators of the x-position of the base of the lane lines. 
    # we can use that as a starting point for where to search for the lines.
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # plt.plot(histogram)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 50
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fit_leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    fit_rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    lane = {}
    lane["left_fit"] = left_fit
    lane["right_fit"] = right_fit
    lane["fit_leftx"] = fit_leftx
    lane["fit_rightx"] = fit_rightx
    lane["fity"] = ploty
    lane["mid_lane"] = (np.max(fit_rightx) - np.min(fit_leftx))
    lane["left"] = np.min(fit_leftx)
    lane["right"] = np.max(fit_rightx)

    return out_img, lane


# In[81]:

# Skip the sliding windows step once we know where the lines are

def detect_next_lane(binary_warped, left_fit, right_fit):

    # We have a new warped binary image from pervious step
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 70
    margin_to_draw = 20
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    # On located the lane line pixels ( x and y pixel positions) fit a second order polynomial curve:
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    fit_leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    fit_rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.zeros_like(binary_warped).astype(np.uint8)
    out_img = np.dstack((out_img, out_img, out_img))*255


    lane = {}
    lane["left_fit"] = left_fit
    lane["right_fit"] = right_fit
    lane["fit_leftx"] = fit_leftx
    lane["fit_rightx"] = fit_rightx
    lane["fity"] = ploty
    lane["mid_lane"] = (np.max(fit_rightx) - np.min(fit_leftx))
    lane["left"] = np.min(fit_leftx)
    lane["right"] = np.max(fit_rightx)

    return out_img, lane


# In[82]:

def radius_of_curvature(image, leftx, rightx, ploty, l, r):

    # define conversion in x and y from pixel space to meters
    y_eval = 719
    ym_per_pix = 30/720
    xm_per_pix = 3.7/(l-r)

    # fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    #calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    display = "\n Radius of curvature (Left): " + str(round(left_curverad,2)) + " m" + "\n Radius of curvature (Right): " + str(round(right_curverad,2)) + " m"

    return display, left_curverad, right_curverad


# In[83]:

# Print output on video stream
def display_info(image, text, mid_lane, l, r):
    
    # Fotn and location to display on the frame
    font = cv2.FONT_HERSHEY_PLAIN
    y0, dy = 20, 20

    img_center = int(image.shape[1] / 2)
    lane_center = int(l + ((r-l)/2))
    xm_per_pix = 3.7/(r-l)
    text = text + "\n Distance from lane center: " + str(round((img_center - lane_center) * xm_per_pix, 2)) + "m" 
    
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(image,line,(20,y), font, 1,(255,255,255),2)
        #cv2.line(image, (lane_center, 720), (lane_center, 680), (255,0,0), 2)
        #cv2.line(image, (img_center, 720), (img_center, 680), (0,0,255), 2) 
    return image


# In[84]:

# Draw the lanes on an empty canvas 
def lanes_warped (warped, left_fitx, right_fitx, ploty):
    # Create an empty image
    warp_zero = np.zeros_like(warped[:,:,2]).astype(np.uint8)
    # three channels
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,150))
    return color_warp


# In[85]:

# Draw the lane on the image of the road
def final_image(image, color_warp):   
    # Inverse perspective matrix (inverse_warp_matrix)
    lanes = cv2.warpPerspective(color_warp, inverse_warp_matrix, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, lanes, 0.39, 0)
    return result


# In[86]:

# If your sanity checks reveal that the lane lines you've detected are problematic
def sanity_check(lane, radius_l, radius_r):

    radii_ok = True
    width_ok = True
    parallel_ok = True

    radius_diff = abs( 1./radius_l - 1./radius_r)
    if (radius_diff > 0.001):
        radii_ok = False

    width = abs(lane["left"] - lane["right"]) # average width of the lane
    if (width < 890 or width > 990):
        width_ok = False

    fit_diff = abs(lane["left_fit"][1] - lane["right_fit"][1])
    if (fit_diff>0.6):
        parallel_ok = False

    return radii_ok and width_ok and parallel_ok


# In[87]:

def pipeline(frame):
    # Left and right lanes
    global last_right_fit
    global last_left_fit
    global warp_matrix
    global inverse_warp_matrix
   
    # Step 1: Undistort frame using camera matrix and distortion co-efficient  
    image = undistort(frame, cam_matrix, dist_coeff)  
    
    # Step 2: apply color and gradient threshold to the image
    image = color_gradient_threshold(image)
    
    # Step 3: Perspective transform from image view to bird (top down) view
    # src and dest points for perspective transform
    src = np.array([[490, 482],[810, 482], [1250, 720],[40, 720]], dtype=np.float32)
    dst = np.array([[0, 0], [1280, 0], [1250, 720],[40, 720]], dtype=np.float32)

    warp_matrix = cv2.getPerspectiveTransform(src, dst)

     # Inverse perspective transformation matrix - helps in ploting output 
    inverse_warp_matrix = cv2.getPerspectiveTransform(dst, src) 
    image = warp(image)
    final = np.copy(image)
    plt.imshow(final)
    
    # Step 4: 
    # Find lane for the first time
    if (len(last_right_fit)==0 and len(last_left_fit)==0):
        image, lane = detect_first_lane(image)
        last_right_fit = lane["right_fit"]
        last_left_fit = lane["left_fit"]
    else:
        # Find next lane
        image, lane = detect_next_lane(image, last_left_fit, last_right_fit)
        txt, radius_l, radius_r = radius_of_curvature(image, lane["fit_leftx"], lane["fit_rightx"], lane["fity"], lane["left"], lane["right"])
        last_right_fit = lane["right_fit"]
        last_left_fit = lane["left_fit"]
        # sanity check and fallback on detect full
        if not sanity_check(lane, radius_l, radius_r):
            img, lane = detect_first_lane(final)
            last_right_fit = lane["right_fit"]
            last_left_fit = lane["left_fit"]

    left_fit = lane["left_fit"]
    right_fit = lane["right_fit"]
    fit_leftx = lane["fit_leftx"]
    fit_rightx = lane["fit_rightx"]
    fity = lane["fity"]
    mid_lane = lane["mid_lane"] 
    left = lane["left"]
    right = lane["right"]
    
    # Step 5: Draw curves on detected lanes
    image = lanes_warped(image, fit_leftx, fit_rightx, fity)

    # Step 6: Get final image - mergre lane and frame
    image = final_image(frame, image)

    # Step 7: Find radius of curvature
    txt, left_curverad, right_curverad = radius_of_curvature(image, fit_leftx, fit_rightx, fity, left, right)

    # Step 8: Calculate lane positions 
    mid_lane = (np.max(fit_rightx) - np.min(fit_leftx))
    left = np.min(fit_leftx)
    right = np.max(fit_rightx)

    # Step 9: Display obtained info on the screen
    image = display_info(image, txt, int(mid_lane), int(left), int(right) )

    return image


# In[88]:


