#include <opencv2/opencv.hpp>
//each filter is considered to be a square filter with linear odd dimension of lsize (3x3 5x5 7x7 and so on)
//each function needs to receive a preallocated filter 2D array with lszie linear dimensions


cv::Mat_<float> hardcodedLGN();
cv::Mat_<float> gaborFilter(int linsize, double sig, double th, double fr, double ph);

