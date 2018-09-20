#include <Filters.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
//each filter is considered to be a square filter with linear odd dimension of lsize (3x3 5x5 7x7 and so on)
//each function needs to receive a preallocated filter 2D array with lszie linear dimensions


cv::Mat_<float> hardcodedLGN()
{   
    unsigned int lsize = 3;
    cv::Mat_<float> kernel(lsize,lsize, CV_32FC1);
    float normal_factor = (lsize*lsize) - 1;
    for(int i = 0; i<lsize; i++){
        for(int j = 0; j<lsize; j++){
            kernel.at<float>(i,j) = -1 / normal_factor;            
        }
    }

    kernel.at<float>(lsize/2,lsize/2) = 1;
    return kernel;
}

//Filter modelled as a mexican hat, work in progress
cv::Mat_<float> newLGN(int linsize, double sig)
{
    int borders = (linsize - 1)/2;
    double sigma = sig/linsize;
    cv::Mat_<float> kernel(linsize,linsize, CV_32FC1);
    for (int y=-borders; y<=borders; y++)
    {
        for (int x=-borders; x<=borders; x++)
        {
            kernel.at<float>(borders+y,borders+x) = (float)(1/(CV_PI*pow(sigma,2)))*(1-0,5*((pow(x,2)+pow(y,2)))/pow(sigma,2))*exp(-0.5*(pow(x,2)+pow(y,2))/pow(sigma,2));
        }
    }
    return kernel;
}


cv::Mat_<float> gaborFilter(int linsize, double sig, double th, double fr, double ph)
{
    int borders = (linsize-1)/2;
    double theta = th*CV_PI/180;
    double phase = ph*CV_PI/180;
    double del = 2.0/(linsize-1);
    double lmbd = fr;
    double sigma = sig/linsize;
    double x_theta;
    double y_theta;
    cv::Mat_<float> kernel(linsize,linsize, CV_32FC1);
    for (int y=-borders; y<=borders; y++)
    {
        for (int x=-borders; x<=borders; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(borders+y,borders+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + phase);
        }
    }
    return kernel;
}