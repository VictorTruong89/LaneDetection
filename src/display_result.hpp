#ifndef DISPLAY_RESULT_HPP_
#define DISPLAY_RESULT_HPP_

#include "semantic_segmentation_method.hpp"


class VideoResultDisplay
{
public:
	cv::Mat DisplayLane(cv::Mat input_image);

	cv::Mat DisplayObject(cv::Mat input_image);

	cv::Mat DisplayText(cv::Mat input_image, std::string text, std::vector<int>position);

	cv::Mat SaveVideo(cv::Mat input_image);

};


#endif // DISPLAY_RESULT_HPP_