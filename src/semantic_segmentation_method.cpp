#include "semantic_segmentation_method.hpp"

using namespace std;


// IMAGE BLURRING
/**
 *@brief Apply gaussian filter to the input image to denoise it
 *@param inputImage is the frame of a video in which the
 *@param lane is going to be detected
 *@return Blurred and denoised image
 */
cv::Mat Lane_SemanticSegmentation::deNoise(cv::Mat input_image) 
{
    cv::Mat output_image;
    int kernel = 3;
    cv::GaussianBlur(input_image, output_image, cv::Size(kernel, kernel), 0, 0);
    return output_image;
}


cv::Mat Lane_SemanticSegmentation::AbsSobelThreshold(cv::Mat input_image, char sobel_orient, int thres_min, int thres_max)
{
    cv::Mat image_sobel, image_sobel_abs;
    if (sobel_orient == 'x')
    {
        cv::Sobel(input_image, image_sobel, CV_64F, 1, 0);
    }
    else cv::Sobel(input_image, image_sobel, CV_64F, 0, 1);
    image_sobel = cv::abs(image_sobel);

    double max, min;
    cv::minMaxIdx(image_sobel, &min, &max);
    cv::convertScaleAbs(image_sobel, image_sobel_abs, (255/max), 0);

    // Create binary mask where gradient thresholds are met
    // By comparing sobel_matrix to a scalar, the return is either 0 / 255 
    cv::Mat mask = ((image_sobel_abs >= thres_min) & (image_sobel_abs <= thres_max));

    return mask;
}


cv::Mat Lane_SemanticSegmentation::DirectionThreshold(cv::Mat input_image, double thres_min, double thres_max)
{
    // Take the gradient by Sobel thresholding in x & y separately
    cv::Mat sobel_x, sobel_y;
    int sorbel_kernel = 3;
    cv::Sobel(input_image, sobel_x, CV_64F, 1, 0, sorbel_kernel);
    cv::Sobel(input_image, sobel_y, CV_64F, 0, 1, sorbel_kernel);

    // Take absolute values of sobel_x & sobel_y gradients
    sobel_x = cv::abs(sobel_x);
    sobel_y = cv::abs(sobel_y);

    // Use cv::phase (equivalent to np.arctan2) to calculate the direction of the gradient
    cv::Mat direction;
    cv::phase(sobel_y, sobel_x, direction, false);
    direction = cv::abs(direction);

    // Create binary mask where directional thresholds are met
    // By comparing sobel_matrix to a scalar, the return is either 0 / 255 
    cv::Mat mask = ((direction >= thres_min) & (direction <= thres_max));

    return mask;
}


cv::Mat Lane_SemanticSegmentation::getThresholdImage(cv::Mat input_image)
{
    // By default OpenCV load COLOR images as BGR (instead of RGB)
    // IMREAD_COLOR --> B/G/R.......This has been tested for this program
    cv::Mat image_gray;
    cv::cvtColor(input_image, image_gray, cv::COLOR_BGR2GRAY);

    // Apply gradient threshold on the horizontal gradient
    cv::Mat gradient_binary_mask = AbsSobelThreshold(image_gray, 'x', 10, 200);

    // Apply directional threshold so that only edges closer to vertical are detected
    cv::Mat direction_binary_mask = DirectionThreshold(image_gray, M_PI / 6, M_PI / 2);

    // Combine the gradient & directional thresholds
    cv::Mat grad_dir_binary_mask = gradient_binary_mask & direction_binary_mask;

    // Split the RGB input_image into separate R/G/B channels
    cv::Mat RGB_channel[3];
    cv::split(input_image, RGB_channel);

    // Filter with R & G channels so that yellow lanes are well detected
    int yellow_lane_threshold = 150;  // WHY "threshold == 150" ???
    cv::Mat R_binary_mask = (RGB_channel[2] > yellow_lane_threshold);
    
    cv::Mat G_binary_mask = (RGB_channel[1] > yellow_lane_threshold);
    
    cv::Mat R_G_binary_mask = R_binary_mask & G_binary_mask;

    // Convert the RGB input_image to HLS & Split it into H/L/S channels
    cv::Mat hls_image;
    cv::cvtColor(input_image, hls_image, cv::COLOR_BGR2HLS);
    cv::Mat hls_channel[3];
    cv::split(hls_image, hls_channel);

    // Filter with S channel to detect bright-yellow & white lanes
    int S_thres_min = 100;   // WHY 100 ???
    int S_thres_max = 255;  // WHY 255 ???
    cv::Mat S_binary_mask = ((hls_channel[2] > S_thres_min) & (hls_channel[2] <= S_thres_max));

    // Filter with L channel to avoid pixels that are darker due to shadows
    int L_thres_min = 120;  // WHY 120 ???
    int L_thres_max = 255;  // WHY 100 ???
    cv::Mat L_binary_mask = ((hls_channel[1] > L_thres_min) & (hls_channel[1] <= L_thres_max));

    /*
    By combining all binary_masks above, a pixel should be either a yellowish or whitis lane
    And it should have a gradient as per our thershold
    */
    cv::Mat combined_threshold = ((R_G_binary_mask & L_binary_mask) & (S_binary_mask | grad_dir_binary_mask));

    return combined_threshold;
}


cv::Mat Lane_SemanticSegmentation::applyROI(cv::Mat input_image)
{
    // Vertices of the Region Of Interest (ROI)
    int height = input_image.size().height;
    int width = input_image.size().width;
    cv::Point roi_vertices[3] = {
        cv::Point(0, height - 1),
        cv::Point(width / 2, height / 2),
        cv::Point(width - 1, height - 1)
    };
    
    // Create a mask of ROI
    cv::Mat roi_mask = cv::Mat::zeros(input_image.size(), input_image.type());
    cv::fillConvexPoly(roi_mask, roi_vertices, 3, cv::Scalar(255, 0 ,0));

    // Multiply the combined_binary_mask & polygonal_mask to get the final_mask
    cv::Mat final_mask;
    cv::bitwise_and(input_image, roi_mask, final_mask);

    return final_mask;
}


cv::Mat Lane_SemanticSegmentation::PerspectiveTransform(cv::Mat input_image, cv::Mat* transform_mat, cv::Mat* transform_mat_inv)
{
    int height = input_image.size().height;
    int width = input_image.size().width;

    // Define calibration box in source & destination (warped) coordinates
    cv::Point2f coordinates_src[4] = {
        cv::Point(280, 700),    // Bottom left
        cv::Point(595, 460),    // Top left
        cv::Point(725, 460),    // Top right
        cv::Point(1125, 700)    // Bottom right
    };

    cv::Point2f coordinates_dst[4] = {
        cv::Point(250, 720),    // Bottom left
        cv::Point(250, 0),      // Top left
        cv::Point(1065, 0),     // Top right
        cv::Point(1065, 720)    // Bottom right
    };

    // Compute the perspective transform matrix
    cv::Mat perspective_transform = cv::getPerspectiveTransform(coordinates_src, coordinates_dst);

    // Compute the inverse perspective transform matrix by swapping the input coordinates
    cv::Mat perspective_transform_inv = cv::getPerspectiveTransform(coordinates_dst, coordinates_src);

    // Warp the image by using linear interpolation (cv::INTER_LINEAR is by default)
    cv::Mat warp_image;
    cv::warpPerspective(input_image, warp_image, perspective_transform, warp_image.size(), cv::INTER_LINEAR);

    return warp_image;
}


void Lane_SemanticSegmentation::FindBaseLanes(cv::Mat input_image, int* lane_left, int* lane_right)
{
    int width = input_image.size().width;
    int height = input_image.size().height;

    cv::Mat left_img = input_image(cv::Range(1, height), cv::Range(1, width / 2));
    cv::Mat right_img = input_image(cv::Range(1, height), cv::Range(width / 2, width));
    
    //std::vector<cv::Point2f> lane;
    // Find all non-zero pixels in the input_image, which is likely to be 
    //cv::findNonZero(input_image, lane);
    
    std::vector<int> hist_left = { 0 };
    std::vector<int> hist_right = { 0 };

    // Find the left_lane starting location in left image
    for (int i = 1; i < left_img.size().width; i++)
    {
        for (int j = (left_img.size().height * 3 / 4); j < left_img.size().height; j++)
        {
            // Mindful that image coordinate is opposite of a typical matrix[row, column]
            // Run the test_img_coordinate() below to review & recheck this
            hist_left[i] = hist_left[i] + left_img.at<int>(j, i);
        }

        // Start comparing the histogram since i == 2 onwards
        if (i > 2)
        {
            if (hist_left[i] > hist_left[i - 1])
            {
                *lane_left = i;
            }
        }
    }

    // Find the right_lane starting location in left image
    for (int i = 1; i < right_img.size().width; i++)
    {
        for (int j = (right_img.size().height * 3 / 4); j < right_img.size().height; j++)
        {
            // Mindful that image coordinate is opposite of a typical matrix[row, column]
            // Run the test_img_coordinate() below to review & recheck this
            hist_right[i] = hist_right[i] + right_img.at<int>(j, i);
        }

        // Start comparing the histogram since i == 2 onwards
        if (i > 2)
        {
            if (hist_right[i] > hist_right[i - 1])
            {
                *lane_right = width / 2 + i;
            }
        }
    }
}


cv::Mat Lane_SemanticSegmentation::DetectLine(cv::Mat input_image, int num_windows, bool display)
{
    int lane_base_left = 1;
    int lane_base_right = 1;
    FindBaseLanes(input_image, &lane_base_left, &lane_base_right);
    // Define the lane_current w.r.t each sliding window
    int lane_current_left = lane_base_left;
    int lane_current_right = lane_base_right;

    // Define the size of the sliding windows
    const int window_width = 200;
    const int window_height = input_image.size().height / num_windows;
    // Minimum number of lane_pixel, above which to relocate the sliding_window
    const int num_pixel = 50;

    // Find all non-zero pixel in the entire input_image
    cv::Mat non_zero_mat;
    cv::findNonZero(input_image, non_zero_mat);
    

    // Create empty vector to receive left & right lane pixel indices
    std::vector<cv::Point> lane_index_left;
    std::vector<cv::Point> lane_index_right;

    // Loop through the windows one-by-one
    for (int window = 0; window < num_windows; window++)
    {
        // Locate the 4 corners of each sliding window left & right
        int win_y_low = input_image.size().height - (window + 1) * window_height;
        int win_y_high = input_image.size().height - window * window_height;
        int win_xleft_low = lane_current_left - window_width / 2;
        int win_xleft_high = lane_current_left - window_width / 2;
        int win_xright_low = lane_current_right - window_width / 2;
        int win_xright_high = lane_current_right - window_width / 2;
        cv::Point upper_left = (win_xleft_high, win_y_high);
        cv::Point lower_left = (win_xleft_low, win_y_low);
        cv::Point upper_right = (win_xright_high, win_y_high);
        cv::Point lower_right = (win_xright_low, win_y_low);

        // Display the sliding window in the perspective transform warped image
        if (display == true)
        {
            cv::rectangle(input_image, upper_left, lower_left, (0, 255, 0), 3);
            cv::rectangle(input_image, upper_right, lower_right, (0, 255, 0), 3);
        }

        // Identify the non-zero pixels in each sliding window
        std::vector<cv::Point> lane_window_left, lane_window_right;
        for (int i = 0; i < non_zero_mat.total(); i++)
        {
            if ((non_zero_mat.at<cv::Point>(i).y > win_y_low) && (non_zero_mat.at<cv::Point>(i).y < win_y_high))
            {
                if ((non_zero_mat.at<cv::Point>(i).x > win_xleft_low) && (non_zero_mat.at<cv::Point>(i).x < win_xleft_high))
                {
                    lane_window_left.push_back(non_zero_mat.at<cv::Point>(i));
                }
                if ((non_zero_mat.at<cv::Point>(i).x > win_xright_low) && (non_zero_mat.at<cv::Point>(i).x < win_xright_high))
                {
                    lane_window_right.push_back(non_zero_mat.at<cv::Point>(i));
                }
            }
        }

        // If enough of non-zero lane_pixel are detected inside each window, recenter the next sliding_window
        /*if (lane_window_left.size() > num_pixel)
        {
            lane_current_left = cv::mean(lane_window_left)[1];
        }
        if (lane_window_right.size() > num_pixel)
        {
            lane_current_right = cv::mean(lane_window_right)[1];
        }*/
    }

    return input_image;
}


int test_img_coordinate()
{
    int width = 300;
    int height = 200;
    
    cv::Mat black = cv::Mat::ones(cv::Size(width, height), CV_8U);
    
    cv::imshow("black", black);
    for (int i = 1; i < width / 2; i++)
    {
        for (int j = height * 3 / 4; j < height; j++)
        {
            black.at<uchar>(j, i) = black.at<uchar>(j, i) * 255;
        }
    }
    cv::imshow("new black", black);
    cv::waitKey(0);

    return 0;
}