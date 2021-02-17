#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered (Harris 2x2 covariance matrix)
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output to 8 bit with convertScaleAbs
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // Locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    // Keypoint candidates have a higher response than the minimum
    vector<cv::KeyPoint> keypoints;
    float kpDiam = 2 * apertureSize; // diameter is twice the Sobel operator apertur
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (int r=0; r<dst_norm.rows; r++)
    {
        for (int c=0; c<dst_norm.cols; c++)
        {
            float resp = dst_norm.at<float>(r,c);
            if (resp >= minResponse)
            {
                cv::KeyPoint kp(c, r, kpDiam, resp);

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool overlap = false;
                // Iterate through existing keypoints
                for (auto it = keypoints.begin(); it != keypoints.end(); it++)
                {   
                    // If there is an overlap with an existing keypoint
                    if (cv::KeyPoint::overlap(kp, *it) > maxOverlap)
                    {
                        overlap = true;
                        // If new keypoint has highest response, replace old one and stop comparison
                        if (resp > it->response)
                        {
                            *it = kp;
                            break;
                        }
                    }
                }
                // If there was no overlap, add keypoint in vector
                if (!overlap)
                    keypoints.push_back(kp);
            }
        }
    }


    cv::Mat keyPointImg = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, keyPointImg);

    // visualize results
    windowName = "Harris Corner Detected Keypoints";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, keyPointImg);
    cv::waitKey(0);
}

int main()
{
    cornernessHarris();
}