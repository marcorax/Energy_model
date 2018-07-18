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
    int posx = 0;
    int posy = 0;
    int checkcounter = 0 ;
    unsigned short test1;
    unsigned char test2;
    cv::Mat M(YDIM, XDIM, CV_32F);

    DAVISFrames testobj(TEST_FILE, XDIM, YDIM);
    std::cout<< testobj.frames.size()<<" extracted Frames."<<std::endl;
    std::cin>>picture;

    for(int i = 0; i<XDIM*YDIM;i++){
        M.at<float>(posy,posx)=((float)testobj.frames[picture][i])/(float)65536;
        test1 = testobj.frames[picture][i];
        test2 = M.at<float>(posy,posx);
        posx++;
        if(posx==XDIM){
            posx=0;
            posy++;
        }
        checkcounter++;
    }
    //std::cout<<"M="<<M<<std::endl;
    //std::cout<<"Last value:"<<testobj.frames[picture][checkcounter-1]<<std::endl;

    if ( !M.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", M);
    cv::waitKey(0);

    return 0;
}

