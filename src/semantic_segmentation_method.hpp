#ifndef SEMANTIC_SEGMENTATION_METHOD_HPP_
#define SEMANTIC_SEGMENTATION_METHOD_HPP_
#define _USE_MATH_DEFINES

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>

#include "Eigen/Dense"
#include "Eigen/Geometry"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/base.hpp>
#include "opencv2/opencv.hpp"

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h> // Include Utilities API


class Lane_SemanticSegmentation
{
public:

	cv::Mat deNoise(cv::Mat input_imgage);

	cv::Mat AbsSobelThreshold(cv::Mat input_image, char sobel_orient, int thres_min, int thres_max);

	cv::Mat DirectionThreshold(cv::Mat input_image, double thres_min, double thres_max);

	cv::Mat applyROI(cv::Mat input_image);

	cv::Mat getThresholdImage(cv::Mat input_image);

	cv::Mat PerspectiveTransform(cv::Mat input_image, cv::Mat* transform_mat, cv::Mat* transform_mat_inv);

	void FindBaseLanes(cv::Mat input_image, int* lane_left, int* lane_right);

	cv::Mat DetectLine(cv::Mat input_image, int num_windows, bool display);

	//cv::Mat SlidingWindow(cv::Mat input_image);



};


#endif  // SEMANTIC_SEGMENTATION_METHOD_HPP_