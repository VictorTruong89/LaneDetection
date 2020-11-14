import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
import numpy as np
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# % matplotlib inline

# list of test images' paths
test_in_paths = glob.glob('test_images/*')


def plot_result(imgs, names, rows=0, cols=0):
    '''
    plot the result
    '''
    if (len(imgs) == 0 or len(names) == 0):
        return -1

    f, ax = plt.subplots(rows, cols, figsize=(16, 8))
    f.tight_layout()
    i = 0
    if rows <= 1:
        for c in range(cols):
            ax[c].imshow(imgs[i], cmap='gray')
            ax[c].set_title('{}'.format(names[i]), fontsize=24)
            ax[c].axis('off')
            i += 1
    else:
        for r in range(rows):
            for c in range(cols):
                ax[r, c].imshow(imgs[i], cmap='gray')
                ax[r, c].set_title('{}'.format(names[i]), fontsize=24)
                ax[r, c].axis('off')
                i += 1
    plt.suptitle(p.split('/')[-1], fontsize=36)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def camera_calibration(cal_path, debug=False):
    '''
    this routine performs camera calibration
    it returns `mtx` and `dist` needed to
    undistort images taken from this camera
    '''
    # list all calibration images paths
    cal_images_names = glob.glob(cal_path)

    # chessboard-specific parameters
    nx = 9
    ny = 6

    # code below is based on classroom example
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    # (x,y,z): (0,0,0), (1,0,0), etc
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates, z stays 0

    for fname in cal_images_names:
        # read in image
        img = cv2.imread(fname)

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # in case chessboard was found successfully
        # it skips 3 images that do not show full chessboard (1, 4 and 5)
        if ret == True:
            # image points will be different for each calibration image
            imgpoints.append(corners)
            # object points are the same for all calibration images
            objpoints.append(objp)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            if debug:
                plt.figure(figsize=(15, 10))
                plt.imshow(img)

    # calibration parameters calculation
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None, None)

    # will only use `mtx` and `dist` in this project, hence return
    return mtx, dist


def undistort_image(image, mtx, dist):
    '''
    returns an undistorted image (after camera calibration)
    '''
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    return dst


def gaussian_blur(image, kernel=5):
    '''
    this routine applies blur to reduce noise in images
    '''
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return blurred


def get_thresholded_image(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # apply gradient threshold on the horizontal gradient
    sx_binary = abs_sobel_thresh(gray, 'x', 10, 200)

    # apply gradient direction threshold so that only edges closer to vertical are detected.
    dir_binary = dir_threshold(gray, thresh=(np.pi / 6, np.pi / 2))

    # combine the gradient and direction thresholds.
    combined_condition = ((sx_binary == 1) & (dir_binary == 1))

    # R & G thresholds so that yellow lanes are detected well.
    color_threshold = 150
    R = img[:, :, 0]
    G = img[:, :, 1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)

    # color channel thresholds
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    L = hls[:, :, 1]

    # S channel performs well for detecting bright yellow and white lanes
    s_thresh = (100, 255)
    s_condition = (S > s_thresh[0]) & (S <= s_thresh[1])

    # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.
    l_thresh = (120, 255)
    l_condition = (L > l_thresh[0]) & (L <= l_thresh[1])

    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    color_combined[(r_g_condition & l_condition) & (s_condition | combined_condition)] = 1

    # apply the region of interest mask
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([[0, height - 1], [width / 2, int(0.5 * height)], [width - 1, height - 1]],
                                           dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(color_combined, mask)

    return thresholded


def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    max_value = np.max(abs_sobel)
    binary_output = np.uint8(255 * abs_sobel / max_value)
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    direction = np.absolute(direction)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mask


def warp(img, src_coordinates=None, dst_coordinates=None):
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])

    if src_coordinates is None:
        src_coordinates = np.float32(
            [[280, 700],  # Bottom left
             [595, 460],  # Top left
             [725, 460],  # Top right
             [1125, 700]])  # Bottom right

    if dst_coordinates is None:
        dst_coordinates = np.float32(
            [[250, 720],  # Bottom left
             [250, 0],  # Top left
             [1065, 0],  # Top right
             [1065, 720]])  # Bottom right

    # Compute the perspective transfor, M
    M = cv2.getPerspectiveTransform(src_coordinates, dst_coordinates)

    # Compute the inverse perspective transfor also by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst_coordinates, src_coordinates)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


'''
CAMERA CALIBRATION
'''
# camera calibration step
mtx, dist = camera_calibration('./camera_cal/calibration*.jpg', debug=True)

# Apply distortion correction to raw images
# Raw Test image
p = test_in_paths[0]
image = mpimg.imread(p)

# camera calibration raw image
ca_img = mpimg.imread('camera_cal/calibration1.jpg')
undistorted_ca = undistort_image(ca_img, mtx, dist)

# STEP 1: UNDISTORT (using camera calibration step matrix and dist)
undistorted = undistort_image(image, mtx, dist)

# Plot the result
imgs = [ca_img, undistorted_ca, image, undistorted]
names = ['Original', 'Undistored', '', '']
plot_result(imgs, names, 2, 2)

'''
USE COLOR TRANSFORMS, GRADIENTS ETC. TO CREATE A THRESHOLDED BINARY IMAGE
'''
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
h = hls[:, :, 0]
l = hls[:, :, 1]
s = hls[:, :, 2]

# Plot the result

imgs = [image, gray, red, h, green, l, blue, s]
names = ['Original', 'Gray', 'R', 'H', 'G', 'L', 'B', 'S']
plot_result(imgs, names, 4, 2)

# STEP 2: GAUSSIAN BLUR
blurred = gaussian_blur(undistorted, kernel=3)
# STEP 3: APPLY COLOR SPACE TRANSFORM AND SOBEL THRESHOLDING
combined = get_thresholded_image(blurred)

# Plot the result
imgs = [image, undistorted, blurred, combined]
names = ['Original', 'Undistored', 'Blurred', 'Color/Sobel Threshold']
plot_result(imgs, names, 2, 2)

'''
APPLY PERSPECTIVE TRANSFORM TO RECTIFY BINARY IMAGE INTO "BIRD-EYE VIEW"
'''
# Run the function

src_coordinates = np.float32(
    [[280, 700],  # Bottom left
     [595, 460],  # Top left
     [725, 460],  # Top right
     [1125, 700]])  # Bottom right

dst_coordinates = np.float32(
    [[250, 720],  # Bottom left
     [250, 0],  # Top left
     [1065, 0],  # Top right
     [1065, 720]])  # Bottom right

coordinates = [src_coordinates, dst_coordinates]

# STEP 4: WARP BINARY IMAGE INTO TOP-DOWN VIEW
warped, M, Minv = warp(image, src_coordinates, dst_coordinates)

# Plot the result
imgs = [image, warped]
names = ['Original', 'Warped (Birds-eye View)']

f, ax = plt.subplots(1, 2, figsize=(16, 8))
f.tight_layout()
i = 0
for c in range(2):
    ax[c].imshow(imgs[i], cmap='gray')
    ax[c].plot(Polygon(coordinates[i]).get_xy()[:, 0], Polygon(coordinates[i]).get_xy()[:, 1], color='red')
    warped, M, Minv = warp(combined)
    ax[c].set_title('{}'.format(names[i]), fontsize=24)
    ax[c].axis('off')
    i += 1

plt.suptitle(p.split('/')[-1], fontsize=36)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

'''
DETECT LANE PIXELS N FIT TO FIND THE LANE BOUNDARY
'''


# Histogram
def find_histogram_peaks(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # Peak in the first half indicates the likely position of the left lane
    half_width = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:half_width])

    # Peak in the second half indicates the likely position of the right lane
    rightx_base = np.argmax(histogram[half_width:]) + half_width

    return histogram, leftx_base, rightx_base


histogram, leftx_base, rightx_base = find_histogram_peaks(warped)

print(leftx_base, rightx_base)
plt.plot(histogram)


# Sliding window search
def detect_lines(img, return_img=False):
    # Take a histogram of the bottom half of the image

    histogram, leftx_base, rightx_base = find_histogram_peaks(img)

    if return_img:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(img.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if return_img:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 3)

            # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    if return_img:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Draw left and right lines
        for index in range(img.shape[0]):
            cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 3, (255, 255, 0))
            cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 3, (255, 255, 0))

        return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty), out_img

    return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty)


# Run the function
lines_fit, left_points, right_points, out_img = detect_lines(warped, return_img=True)

# Plot the result
imgs = [warped, out_img]
names = ['Warped image', 'Lane lines detected']
plot_result(imgs, names, 1, 2)


# Searching aroung the previously detected lane
def get_averaged_line(previous_lines, new_line):
    '''
        This function computes an averaged lane line by averaging over previous good frames.
    '''

    # Number of frames to average over
    num_frames = 12

    if new_line is None:
        # No line was detected

        if len(previous_lines) == 0:
            # If there are no previous lines, return None
            return previous_lines, None
        else:
            # Else return the last line
            return previous_lines, previous_lines[-1]
    else:
        if len(previous_lines) < num_frames:
            # we need at least num_frames frames to average over
            previous_lines.append(new_line)
            return previous_lines, new_line
        else:
            # average over the last num_frames frames
            previous_lines[0:num_frames - 1] = previous_lines[1:]
            previous_lines[num_frames - 1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += previous_lines[i]
            new_line /= num_frames
            return previous_lines, new_line


def detect_similar_lines(img,
                         line_fits=None,
                         past_good_left_lines=[],
                         past_good_right_lines=[],
                         running_mean_difference_between_lines=0,
                         return_img=False):
    if line_fits is None:
        return detect_lines(img, return_img)

    left_fit = line_fits[0]
    right_fit = line_fits[1]

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If any of the lines could not be found,
    # perform a more exhaustive search
    if (leftx.size == 0 or rightx.size == 0):
        return detect_lines(img, return_img)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting

    # If no pixels were found return None
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Smoothing
    mean_difference = np.mean(right_fitx - left_fitx)

    if running_mean_difference_between_lines == 0:
        running_mean_difference_between_lines = mean_difference

    if (
            mean_difference < 0.7 * running_mean_difference_between_lines or mean_difference > 1.3 * running_mean_difference_between_lines):
        if len(past_good_left_lines) == 0 and len(past_good_right_lines) == 0:
            return detect_lines(img, return_img)
        else:
            left_fitx = past_good_left_lines[-1]
            right_fitx = past_good_right_lines[-1]
    else:
        past_good_left_lines, left_fitx = get_averaged_line(past_good_left_lines, left_fitx)
        past_good_right_lines, right_fitx = get_averaged_line(past_good_right_lines, right_fitx)
        mean_difference = np.mean(right_fitx - left_fitx)
        running_mean_difference_between_lines = 0.9 * running_mean_difference_between_lines + 0.1 * mean_difference

    if return_img:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        for index in range(img.shape[0]):
            cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 3, (255, 255, 0))
            cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 3, (255, 255, 0))

        return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty), out_img

    return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty)


# Run the function
# Notice I am passing the same image than before.
# In a video stream, it should be passed the next frame.
lines_fit, left_points, right_points, out_img = detect_similar_lines(warped, lines_fit, return_img=True)

# Plot the result
imgs = [warped, out_img]
names = ['Warped image', 'Lane lines detected']
plot_result(imgs, names, 1, 2)

'''
DETERMINE THE CURVATURE OF THE LANE & VEHICLE POSITION W.R.T. CENTER OF 2 LANES
'''


def curvature_radius(leftx, rightx, img_shape, xm_per_pix=3.7 / 800, ym_per_pix=25 / 720):
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 800  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)


# STEP 8: CALCULATE OFFSET DISTANCE AND ROAD CURVATURE

# Run the function
curvature_rads = curvature_radius(leftx=left_points[0], rightx=right_points[0], img_shape=image.shape)

# Print the results
print('Left line curvature:', curvature_rads[0], 'm')
print('Right line curvature:', curvature_rads[1], 'm')


def car_offset(leftx, rightx, img_shape, xm_per_pix=3.7 / 800):
    ## Image mid horizontal position
    mid_imgx = img_shape[1] // 2

    ## Car position with respect to the lane
    car_pos = (leftx[-1] + rightx[-1]) / 2

    ## Horizontal car offset
    offsetx = (mid_imgx - car_pos) * xm_per_pix

    return offsetx


# Run the function
offsetx = car_offset(leftx=left_points[0], rightx=right_points[0], img_shape=image.shape)

print('Car offset from center:', offsetx, 'm.')

'''
WARP THE DETECTED LANE BOUNDARY BACK ONTO THE ORIGINAL IMAGE
'''


# STEP 7: UNWARP OVERLAY FROM TOP-DOWN VIEW BACK INTO CAMERA VIEW
def draw_lane(img, warped_img, left_points, right_points, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fitx = left_points[0]
    right_fitx = right_points[0]
    ploty = left_points[1]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)


# Run the function
img_lane = draw_lane(image, warped, left_points, right_points, Minv)

# Plot the result
imgs = [image, img_lane]
names = ['Original', 'Lane detected']
plot_result(imgs, names, 1, 2)

'''
OUTPUT VISUAL DISPLAY OF THE LANE BOUNDARY & 
NUMERICAL ESTIMATION OF THE LANE CURVATURE & VEHICLE POSITION
'''


def add_metrics(img, leftx, rightx, xm_per_pix=3.7 / 800, ym_per_pix=25 / 720):
    # Calculate radius of curvature
    curvature_rads = curvature_radius(leftx=leftx, rightx=rightx, img_shape=img.shape,
                                      xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)
    # Calculate car offset
    offsetx = car_offset(leftx=leftx, rightx=rightx, img_shape=img.shape)

    # Display lane curvature
    out_img = img.copy()
    cv2.putText(out_img, 'Left lane line curvature: {:.2f} m'.format(curvature_rads[0]),
                (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(out_img, 'Right lane line curvature: {:.2f} m'.format(curvature_rads[1]),
                (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    # Display car offset
    cv2.putText(out_img, 'Horizontal car offset: {:.2f} m'.format(offsetx),
                (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    return out_img


# Run the function
out_img = add_metrics(img_lane, leftx=left_points[0], rightx=right_points[0])

# Plot the result
imgs = [image, out_img]
names = ['Original', 'Lane detected with metrics']
plot_result(imgs, names, 1, 2)

'''
RUN THE PIPELINE IN A VIDEO
'''


class Pipeline:
    def __init__(self, cal_path):
        # Make a list of calibration images

        # Calibrate camera
        self.mtx, self.dist = camera_calibration(cal_path)

        # Reinitialize
        self.lines_fit = None
        self.past_good_left_lines = []
        self.past_good_right_lines = []
        self.running_mean_difference_between_lines = 0

    def __call__(self, img):
        # STEP 1: UNDISTORT (using camera calibration step matrix and dist)
        undistorted = undistort_image(img, self.mtx, self.dist)

        # STEP 2: GAUSSIAN BLUR
        blurred = gaussian_blur(undistorted, kernel=3)

        # STEP 3: APPLY COLOR SPACE TRANSFORM AND SOBEL THRESHOLDING
        combined = get_thresholded_image(blurred)

        # Apply a perspective transform to rectify binary image ("birds-eye view")
        src_coordinates = np.float32(
            [[280, 700],  # Bottom left
             [595, 460],  # Top left
             [725, 460],  # Top right
             [1125, 700]])  # Bottom right

        dst_coordinates = np.float32(
            [[250, 720],  # Bottom left
             [250, 0],  # Top left
             [1065, 0],  # Top right
             [1065, 720]])  # Bottom right

        # STEP 4: WARP BINARY IMAGE INTO TOP-DOWN VIEW
        warped, M, Minv = warp(combined, src_coordinates, dst_coordinates)

        # STEP 5: Detect lane pixels and fit to find the lane boundary
        self.lines_fit, left_points, right_points, out_img = detect_similar_lines(warped,
                                                                                  self.lines_fit,
                                                                                  self.past_good_left_lines,
                                                                                  self.past_good_right_lines,
                                                                                  self.running_mean_difference_between_lines,
                                                                                  return_img=True, )

        # STEP 6: Warp the detected lane boundaries back onto the original image.
        img_lane = draw_lane(img, warped, left_points, right_points, Minv)

        # STEP 7: Add metrics to the output img
        out_img = add_metrics(img_lane, leftx=left_points[0], rightx=right_points[0])

        return out_img


img = mpimg.imread('test_images/test2.jpg')

# Process video frames with our 'process_image' function
process_image = Pipeline('./camera_cal/calibration*.jpg')

# Apply pipeline
processed = process_image(img)

# Plot the 2 images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(processed, cmap='gray')
ax2.set_title('Processed Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def apply_video(input_video, output_video):
    # Process video frames with our 'process_image' function
    process_image = Pipeline('./camera_cal/calibration*.jpg')

    ## You may uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip(input_video).subclip(0,5)
    clip1 = VideoFileClip(input_video)
    white_clip = clip1.fl_image(process_image)
    # % time
    white_clip.write_videofile(output_video, audio=False)

    print("Apply video pipeline: SUCCESS!")


input_video = './project_video.mp4'
output_video = './project_video_output.mp4'

apply_video(input_video, output_video)

HTML("""
<video width="640" height="360" controls>
  <source src="{0}">
</video>
""".format(output_video))
