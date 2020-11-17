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

	int HistogramCalc(cv::Mat input_image);

	void FindBaseLanes(cv::Mat input_image, int* lane_base);

	void Lane_SemanticSegmentation::DetectLine(cv::Mat input_image, PixelCoordinate* pixel_coordinate, int num_windows, bool display);

	void Lane_SemanticSegmentation::polyfit(const std::vector<double>& xv, const std::vector<double>& yv, std::vector<double>& coeff, int order);

	cv::Mat Lane_SemanticSegmentation::getAverageLine(cv::Mat input_image, int num_windows, bool display);

	//cv::Mat SlidingWindow(cv::Mat input_image);



};


struct PixelCoordinate
{
	// For using OpenCV library
	std::vector<cv::Point> lane_pixel_left;
	std::vector<cv::Point> lane_pixel_right;

	// For using Eigen library
	Eigen::VectorXd left_coordinate_x;
	Eigen::VectorXd left_coordinate_y;
	Eigen::VectorXd right_coordinate_x;
	Eigen::VectorXd right_coordinate_y;

	// Parse pixel_coord from OpenCV to Eigen
	void OpenCV_2_Eigen()
	{
		for (int i = 0; i < lane_pixel_left.size(); i++)
		{
			left_coordinate_x[i] = lane_pixel_left[i].x;
			left_coordinate_y[i] = lane_pixel_left[i].y;
		}
		for (int i = 0; i < lane_pixel_right.size(); i++)
		{
			right_coordinate_x[i] = lane_pixel_right[i].x;
			right_coordinate_y[i] = lane_pixel_right[i].y;
		}
	}

};

#endif  // SEMANTIC_SEGMENTATION_METHOD_HPP_