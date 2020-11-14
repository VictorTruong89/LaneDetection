// REFERENCE: https://github.com/tinachientw/CarND-Advanced-Lane-Lines

#include <iostream>
#include <chrono>
#include <string>



#include "realsense_helper.hpp"
#include "opencv_helper.hpp"
#include "semantic_segmentation_method.hpp"
#include "display_result.hpp"

using namespace std;

cv::Mat read_image()
{
	/*
	INITIALIZE CODING OBJECTS
	*/
	Lane_SemanticSegmentation lane_detector;
	VideoResultDisplay video_display;

	// Location of the input trial video and trial image folders
	string path_video = "D:/CODE/PRIVATE/20201111_lane_detection_semantic_segmentation/video_input/";
	string path_image = "D:/CODE/PRIVATE/20201111_lane_detection_semantic_segmentation/images/test_images/";
	
	// Specific file names of video & image
	string video_name = "project_video.mp4";
	string image_name = "test6.jpg";
	path_video.append(video_name);
	path_image.append(image_name);

	// Create the input video object
	cv::VideoCapture input_video(path_video);
	if (input_video.isOpened() == false)
	{
		std::cout << "Failed to open input video\n";
	}
	
	/*cv::Mat frame;
	while (input_video.isOpened())
	{
		input_video.read(frame);
		cv::imshow("frame", frame);
		if (frame.empty()) break;
		cv::waitKey(1);
	}
	input_video.release();*/

	// OpenCV load image in BGR sequence (instead of RGB elsewhere)
	cv::Mat input_image = cv::imread(path_image, cv::IMREAD_COLOR);
	/*cv::Mat hsl_image;
	cv::cvtColor(input_image, hsl_image, cv::COLOR_RGB2HLS);
	cv::Mat split_image[3];
	cv::split(hsl_image, split_image);
	cv::imshow("hsl combined", hsl_image);
	cv::imshow("h_channel", split_image[0]);
	cv::imshow("s_channel", split_image[1]);
	cv::imshow("l_channel", split_image[2]);
	cv::waitKey(0);*/

	return input_image;
}




int main()
{
	cv::Mat input_image = read_image();

	int height = input_image.size().height;
	int width = input_image.size().width;

	/*cv::Mat left_img = input_image(cv::Range(1, height), cv::Range(1, width / 2));
	cv::Mat right_img = input_image(cv::Range(1, height), cv::Range(width / 2, width));
	cv::imshow("left", left_img);
	cv::imshow("right", right_img);*/
	
	
	Lane_SemanticSegmentation lane_detector;

	cv::Mat deNoise = lane_detector.deNoise(input_image);
	cv::Mat thresholded = lane_detector.getThresholdImage(deNoise);
	cv::Mat masked = lane_detector.applyROI(thresholded);

	// Find lanes
	cv::imshow("original", input_image);
	cv::imshow("denoise", deNoise);
	cv::imshow("thresholded", thresholded);
	cv::imshow("masked", masked);

	// Perspective transform
	cv::Mat perspective_transform_mat, perspective_transform_mat_inv;
	cv::Mat bird_eye = lane_detector.PerspectiveTransform(masked, &perspective_transform_mat, &perspective_transform_mat_inv);
	cv::imshow("bird_eye", bird_eye);
	
	// Show sliding windows
	int lane_base_left = 1;
	int lane_base_right = 1;
	//lane_detector.FindBaseLanes(bird_eye, &lane_base_left, &lane_base_right);
	std::cout << "lane left: " << lane_base_left << endl;
	std::cout << "right lane: " << lane_base_right << endl;
	//bird_eye = lane_detector.DetectLine(bird_eye, 10, true);
	//cv::imshow("sliding windows", bird_eye);

	cv::waitKey(0);

	return 0;
}