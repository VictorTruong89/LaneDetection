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


int Lane_SemanticSegmentation::HistogramCalc(cv::Mat input_image)
{
    int width = input_image.size().width;
    int height = input_image.size().height;

    std::vector<cv::Point2f> non_zero;
    cv::findNonZero(input_image, non_zero);
    int base_x_location = 0;

    // Histogram (in this context) is the x_coordinate of most non_zero pixels
    // It is calculated by the mean value of all x_coordinates
    for (int i = 0; i < non_zero.size(); i++)
    {
        base_x_location += non_zero[i].x;
    }
    base_x_location = base_x_location / non_zero.size();

    return base_x_location;
}


void Lane_SemanticSegmentation::FindBaseLanes(cv::Mat input_image, int* lane_base)
{
    int width = input_image.size().width;
    int height = input_image.size().height;

    // We only detect lane at 1/4 bottom of the frame to reduce the problem of overly-curve lane
    cv::Mat half_image[2];    // [0] is left, [1] is right
    for (int side = 0; side < 2; side++)
    {
        // Split the image into 1/2-width & 1/4-bottom to reduce the problem of overly-curve lane
        half_image[side] = input_image(cv::Range(height * 3 / 4, height), cv::Range(side * width / 2, (side + 1) * width / 2));

        // x_coordinate of lane is estimated as where most non_zero pixel located
        lane_base[side] = HistogramCalc(half_image[side]) + side * (width / 2);
    }
}


void Lane_SemanticSegmentation::DetectLine(cv::Mat input_image, PixelCoordinate* pixel_coordinate, int num_windows, bool display)
{
    // First draftly estimate the coordinate of base_lane by looking at entire image
    int window_base[2];     // [0] is left, [1] is right
    //int window_base_right = 1;
    FindBaseLanes(input_image, &window_base[0]);
    cout << "window_base_left x = " << window_base[0] << endl;
    cout << "window_base_right x = " << window_base[1] << endl;

    // Define the size of the sliding windows
    const int window_width = 100;
    const int window_height = input_image.size().height / num_windows;
    
    // Minimum number of lane_pixel, above which to relocate the sliding_window
    const int num_pixel = 100;

    // Create empty cv::Mat to crop the sliding_windows later
    cv::Mat crop_window;     // [0] is left, [1] is right

    // good_lane is again detected as non_zero within the sliding windows alone
    for (int side = 0; side < 2; side++)
    {
        for (int window = 0; window < num_windows; window++)
        {
            // Locate the 4 corners of each sliding window left & right
            // "win_x/y_low" is the top-left corner
            // "win_x/y_high" is the bottom-right corner
            if(window == 0)
                cout << "window_base x2 = " << window_base[side] << endl;
            int win_y_low = input_image.size().height - (window + 1) * window_height;
            int win_y_high = input_image.size().height - window * window_height;
            int win_x_low = window_base[side] - window_width / 2;
            int win_x_high = window_base[side] + window_width / 2;
            // Be mindful the inverse x/y coordinate of OpenCV

            // Crop the input_image by the sliding windows
            cv::Mat slide_window(input_image, cv::Rect(win_x_low, win_y_low, window_width, window_height));
            slide_window.copyTo(crop_window);

            // All non_zero pixels inside these sliding windows supposed to be good_lane_pixel
            std::vector<cv::Point> window_lane_pixel;
            cv::findNonZero(crop_window, window_lane_pixel);

            // We need to adjust the (x, y) of lane_pixel by the sliding_window_coordinate
            int lane_pixel_mean = 0;
            // For left sliding windows
            if (window_lane_pixel.size() > 0)
            {
                for (int i = 0; i < window_lane_pixel.size(); i++)
                {
                    window_lane_pixel[i].x += win_x_low;
                    window_lane_pixel[i].y += win_y_low;

                    // Append the window_lane_pixel into the vector of good_lane_pixel
                    if (side == 0)
                        pixel_coordinate->lane_pixel_left.push_back(window_lane_pixel[i]);
                    else
                        pixel_coordinate->lane_pixel_right.push_back(window_lane_pixel[i]);

                    lane_pixel_mean += window_lane_pixel[i].x;
                }
                lane_pixel_mean = lane_pixel_mean / window_lane_pixel.size();
            }

            // If sufficient non-zero lane_pixel are detected inside each window, recenter the next sliding_window
            if (window_lane_pixel.size() > num_pixel)
            {
                window_base[side] = lane_pixel_mean;
            }
        }
    }
}


void PolynomialDisplay(cv::Mat input_image, std::vector<cv::Point>polyline)
{
    cv::Mat canvas = cv::Mat::zeros(input_image.size(), input_image.type());
    cv::polylines(canvas, polyline, true, cv::Scalar(255));
    cv::imshow("polyline", canvas);
    cv::waitKey(0);
}


// CALCULATE POLINOMIAL COOEFICIENT OF LANE PIXELS
/**
 *@brief Find the cooeficients of the polyline that fit all lane_pixel
 *@param[in] xvals the x_coordinate of lane_pixels
 *@param[in] yvals the y_coordinate of lane_pixels
 *@param[in] order the order of the polinomial function
 *@return Eigen::VectorXd of the cooficients of the polinomial function
 */
Eigen::VectorXd PolynomialFit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) 
{
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);
    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }
    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }
    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}


// PREDICT THE Y_VALS BASE ON THE POLINOMIAL COOEFICIENT
/**
 *@brief Find the cooeficients of the polyline that fit all lane_pixel
 *@param[in] coeffs vector of coeficients of the polinomial function
 *@param[in] x array 
 *@param[in] order the order of the polinomial function
 *@return Eigen::VectorXd of the cooficients of the polinomial function
 */
double PolynomialEval(Eigen::VectorXd coeffs, double x) 
{
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}


cv::Mat Lane_SemanticSegmentation::getAverageLine(cv::Mat input_image, int num_windows, bool display)
{

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