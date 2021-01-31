#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void magnitudeSobel()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1gray.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // apply smoothing using the GaussianBlur() function from the OpenCV
    cv::Mat imgFilt;
    int size = 5;
    float std = 2.;
    cv::GaussianBlur(imgGray, imgFilt, cv::Size(size, size), std);

    // create filter kernels using the cv::Mat datatype both for x and y
    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2, 
                        -1, 0, +1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);
    float sobel_y[9] = {-1, -2, -1,
                         0,  0,  0, 
                        +1, +2, +1};
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);
    
    // apply filter using the OpenCv function filter2D()
    cv::Mat res_x, res_y;
    cv::filter2D(imgFilt, res_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(imgFilt, res_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // compute magnitude image based on the equation presented in the lesson
    cv::Mat magnitude = res_x.clone();
    for (int r = 0; r < res_x.rows; r++) 
    {
        for (int c = 0; c < res_x.cols; c++)
            magnitude.at<unsigned char>(r, c) = sqrt(pow(res_x.at<unsigned char>(r,c), 2) + pow(res_y.at<unsigned char>(r,c), 2));
    }
    // show result
    string windowName = "Gaussian Blurring";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, magnitude);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    magnitudeSobel();
}