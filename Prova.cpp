#define TEST_FILE "/home/marcorax93/Repositories/Energy_model/Test Files/Frames_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define XDIM 240
#define YDIM 180

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#include <iostream>
#include <Davisloading.hpp>
#include <opencv2/opencv.hpp>



int main(){
    int picture;

    DAVISFrames testobj(TEST_FILE, XDIM, YDIM);
    std::cout<< testobj.frames.size()<<" extracted Frames."<<std::endl;
    std::cin>>picture;

    if ( !testobj.frames[picture].data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", testobj.frames[picture]);
    cv::waitKey(0);

    return 0;
}

