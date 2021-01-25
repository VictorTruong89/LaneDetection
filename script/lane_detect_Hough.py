import cv2
import os
import numpy as np


##### TO CHOOSE EITHER IMAGE OR VIDEO 
load_image = False
load_video = True



##### DETECT EDGE FROM FRAME #####
def canny_edge(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 50, 150)    # Why choosing 50 and 150
    return img_canny


##### MASKING THE REGION OF INTEREST #####
def roi_mask(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    print(image.shape)

    # Define the size of ROI
    polygon = np.array([[(200, img_height), (1100, img_height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)

    # Create a mask from the polygon
    img_mask = cv2.bitwise_and(image, mask)
    return img_mask


##### HOUGH TRANSFORM #####
def show_lines(image, lines):
    img_lines = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            X1, Y1, X2, Y2 = line.reshape(4)
            cv2.line(img_lines, (X1, Y1), (X2, Y2), color=(255, 255, 0), thickness=5)
    return img_lines


##### OPTIMIZE THE LANE DETECTION #####
def make_coordinates(image, line_params):
    try:
        slope, intercept = line_params
    except:
        slope, intercept = 0.001, 0
    y1 = image.shape[0] # image height
    y2 = int(y1 * 3/5)
    x1 = int( (y1 - intercept) / slope )
    x2 = int( (y2 - intercept) / slope )
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit( (x1, x2), (y1, y2), 1 )
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append( (slope, intercept) )
        else:
            right_fit.append( (slope, intercept) )
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array( [left_line, right_line] )





##### LOAD IMAGE ######
img_dir = "D:/CODE/PRIVATE/LaneDetection/images/test_images"
img_dir = os.path.join(img_dir, "straight_lines1.jpg")
image = cv2.imread(img_dir)
img_color = np.copy(image)

if load_image == True:
    img_canny = canny_edge(image)
    img_mask = roi_mask(img_canny)
    lane_lines = cv2.HoughLinesP(img_mask, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
    average_lines = average_slope_intercept(image, lane_lines)
    img_lines = show_lines(img_color, average_lines)
    img_combine = cv2.addWeighted(img_color, 0.8, img_lines, 1, 1)
    cv2.imshow('img_lines', img_lines)
    cv2.imshow('img_combine', img_combine)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()



##### LOAD VIDEO ######
vid_dir = "D:/CODE/PRIVATE/LaneDetection/videos"
vid_dir = os.path.join(vid_dir, "test_video.mp4")
if load_video == True:
    cap = cv2.VideoCapture(vid_dir)

    while(cap.isOpened()):
        _, frame = cap.read()
        frame_canny = canny_edge(frame)
        frame_mask = roi_mask(frame_canny)
        lane_lines = cv2.HoughLinesP(frame_mask, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
        average_lines = average_slope_intercept(frame, lane_lines)
        frame_lines = show_lines(frame, average_lines)
        frame_combine = cv2.addWeighted(frame, 0.8, frame_lines, 1, 1)
        cv2.imshow("result", frame_combine)
        #cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()