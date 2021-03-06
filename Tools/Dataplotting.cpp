#include <Dataplotting.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#include <iostream>
#include <Davisloading.hpp>
#include <opencv2/opencv.hpp>


void printDavisStereo(unsigned int framepos, DAVISFrames & frames_l, DAVISFrames & frames_r,
                      DAVISEvents & events_l, DAVISEvents & events_r, unsigned int t_halfspan,
                      int XDIM, int YDIM, const int & verbose){
    

    cv::Mat Newpicl(YDIM, XDIM, CV_32FC3);
    cv::Mat Newpicr(YDIM, XDIM, CV_32FC3);

    
    if ( !frames_l.frames[framepos].data && !frames_r.frames[framepos].data)
    {
        printf("No image data \n");
    }
    else{

    if(verbose){ //if verbose is set 1 (or any other value than 0) then these debugging message can be seen.
        std::cout<<"L Frame at us: "<<frames_l.end_ts[framepos]<<std::endl;
        std::cout<<"R Frame at us: "<<frames_r.end_ts[framepos]<<std::endl;
    }
    
    cv::cvtColor(frames_l.frames[framepos], Newpicl, CV_GRAY2RGB);
    cv::cvtColor(frames_r.frames[framepos], Newpicr, CV_GRAY2RGB);

    for(unsigned int i = 0; i < events_l.polarity.size(); i++){
        if(events_l.timestamp[i]>frames_l.end_ts[framepos]-t_halfspan && events_l.timestamp[i]<frames_l.end_ts[framepos]+t_halfspan){
            if(events_l.polarity[i]==1){ //ON event, red by standards. Btw, [2] = Red, [1] = Green, [0] = Blue. I don't have any clues on why :I 
                Newpicl.at<cv::Vec3f>(events_l.y_addr[i],events_l.x_addr[i])[0] = (float) 0; 
                Newpicl.at<cv::Vec3f>(events_l.y_addr[i],events_l.x_addr[i])[1] = (float) 0; 
                Newpicl.at<cv::Vec3f>(events_l.y_addr[i],events_l.x_addr[i])[2] = (float) 1; //max float value expected by Opencv
            }
            else{   //OFF event, green by standards
                Newpicl.at<cv::Vec3f>(events_l.y_addr[i],events_l.x_addr[i])[0] = (float) 0; 
                Newpicl.at<cv::Vec3f>(events_l.y_addr[i],events_l.x_addr[i])[1] = (float) 1; //max float value expected by Opencv
                Newpicl.at<cv::Vec3f>(events_l.y_addr[i],events_l.x_addr[i])[2] = (float) 0;
            }
        }
    }
     for(unsigned int i = 0; i < events_r.polarity.size(); i++){
        if(events_r.timestamp[i]>frames_r.end_ts[framepos]-t_halfspan && events_r.timestamp[i]<frames_r.end_ts[framepos]+t_halfspan){
            if(events_r.polarity[i]==1){ //ON event, red by standards. Btw, [2] = Red, [1] = Green, [0] = Blue. I don't have any clues on why :I 
                Newpicr.at<cv::Vec3f>(events_r.y_addr[i],events_r.x_addr[i])[0] = (float) 0; 
                Newpicr.at<cv::Vec3f>(events_r.y_addr[i],events_r.x_addr[i])[1] = (float) 0; 
                Newpicr.at<cv::Vec3f>(events_r.y_addr[i],events_r.x_addr[i])[2] = (float) 1; //max float value expected by Opencv
            }
            else{   //OFF event, green by standards
                Newpicr.at<cv::Vec3f>(events_r.y_addr[i],events_r.x_addr[i])[0] = (float) 0; 
                Newpicr.at<cv::Vec3f>(events_r.y_addr[i],events_r.x_addr[i])[1] = (float) 1; //max float value expected by Opencv
                Newpicr.at<cv::Vec3f>(events_r.y_addr[i],events_r.x_addr[i])[2] = (float) 0;
            }
        }
    }
    //Commands for displaying the left image
    cv::namedWindow("Left DAVIS picture: " + std::to_string(framepos), cv::WINDOW_NORMAL);
    cv::resizeWindow("Left DAVIS picture: " + std::to_string(framepos), XDIM,YDIM);
    cv::imshow("Left DAVIS picture: " + std::to_string(framepos), Newpicl);
    //Commands for displaying the right image
    cv::namedWindow("Right DAVIS picture: " + std::to_string(framepos), cv::WINDOW_NORMAL);
    cv::resizeWindow("Right DAVIS picture: " + std::to_string(framepos), XDIM,YDIM);
    cv::imshow("Right DAVIS picture: " + std::to_string(framepos), Newpicr);

    //cv::waitKey(0);
    }
}


/*Function used to print fused rapresentation of on/off events or both negative/positive images, magnifing factor is handy when plotting
small images like when checking filters*/

void printOnOffImages(std::string Windowmsg, cv::Mat_<float> on_image, cv::Mat_<float> off_image, unsigned int magnifingFactor){
    if (on_image.cols != off_image.cols || on_image.rows != off_image.rows){
        std::cout<<"on_image and off_image dimensions not corresponding!"<<std::endl;
        return;
    }
    
    cv::Mat Newpic, BlueChannel;
    
    std::vector<cv::Mat> channels;
    BlueChannel = cv::Mat::zeros(cv::Size(on_image.cols, on_image.rows), CV_32FC1); 

    channels.push_back(BlueChannel);//blue channel
    channels.push_back(off_image);//green channel
    channels.push_back(on_image);//red channel

    
    cv::merge(channels, Newpic);//merge combine a vector of single channel pictures in order to make a multichannel cv mat 
    
    cv::namedWindow(Windowmsg, cv::WINDOW_NORMAL);
    cv::resizeWindow(Windowmsg, on_image.cols*magnifingFactor, on_image.rows*magnifingFactor);
    cv::imshow(Windowmsg, Newpic);
}

void printImage(std::string Windowmsg, const cv::Mat_<float> &image, unsigned int magnifingFactor){
    
    cv::namedWindow(Windowmsg, cv::WINDOW_NORMAL);
    cv::resizeWindow(Windowmsg, image.cols, image.rows);
    cv::imshow(Windowmsg, image);

}