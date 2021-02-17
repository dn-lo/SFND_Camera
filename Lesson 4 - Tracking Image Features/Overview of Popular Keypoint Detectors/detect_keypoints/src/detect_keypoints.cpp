#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void detKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Shi-Tomasi detector
    int blockSize = 6;       //  size of a block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    double qualityLevel = 0.01;                                   // minimal accepted quality of image corners
    double k = 0.04;
    bool useHarris = false;

    vector<cv::KeyPoint> kptsShiTomasi;
    vector<cv::Point2f> corners;
    auto tSta = std::chrono::steady_clock::now();
    cv::goodFeaturesToTrack(imgGray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);
    auto tEnd = std::chrono::steady_clock::now();
  	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tSta);
    cout << "Shi-Tomasi with n = " << corners.size() << " keypoints in " << dt.count() << " ms" << endl;

    for (auto point : corners)
    { // add corners to result vector
        cv::KeyPoint newKeyPoint(point, blockSize);
        kptsShiTomasi.push_back(newKeyPoint);
    }

    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImage);
    cv::waitKey(0);

    // use the OpenCV library to add the FAST detector
    // in addition to the already implemented Shi-Tomasi 
    // detector and compare both algorithms with regard to 
    // (a) number of keypoints, (b) distribution of 
    // keypoints over the image and (c) processing speed.
    int threshold = 30;  // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;                   // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    cv::Ptr<cv::FeatureDetector> fast = cv::FastFeatureDetector::create(threshold, bNMS, type);
    vector<cv::KeyPoint> kptsFast;

    tSta = std::chrono::steady_clock::now();
    fast->detect(imgGray, kptsFast, cv::Mat());
    tEnd = std::chrono::steady_clock::now();
  	dt = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tSta);
    cout << "FAST detector with n = " << kptsFast.size() << " keypoints in " << dt.count() << " ms" << endl;

    // visualize results
    cv::Mat visImageFast = img.clone();
    cv::drawKeypoints(img, kptsFast, visImageFast, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    windowName = "FAST detector Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImageFast);
    cv::waitKey(0);



}

int main()
{
    detKeypoints1();
}