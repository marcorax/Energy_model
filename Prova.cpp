#define AEDAT_FOLDER  "/home/marcorax93/Repositories/Energy_model/TestFiles/"
#define FRAMES_L "Frames_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define EVENTS_L "Events_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define FRAMES_R "Frames_R_Moving_Bar-2018_03_06_17_04_05.aedat"
#define EVENTS_R "Events_R_Moving_Bar-2018_03_06_17_04_05.aedat"

#define VERBOSE 1 //set to 0 di disable, 1 (or any other integer) to enable
#define XDIM 240
#define YDIM 180

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#include <iostream>
#include <Davisloading.hpp>
#include <opencv2/opencv.hpp>
#include <Dataplotting.hpp>



int main(){
    int framepos;
    unsigned int t_halfspan = 1000;

    DAVISFrames frames_l(std::string(AEDAT_FOLDER) + std::string(FRAMES_L), XDIM, YDIM, VERBOSE);
    std::cout<< frames_l.frames.size()<<" extracted Frames from the left camera."<<std::endl;
    DAVISEvents events_l(std::string(AEDAT_FOLDER) + std::string(EVENTS_L), VERBOSE);
    std::cout<< events_l.polarity.size()<<" extracted Events from the left camera."<<std::endl;
    DAVISFrames frames_r(std::string(AEDAT_FOLDER) + std::string(FRAMES_R), XDIM, YDIM, VERBOSE);
    std::cout<< frames_r.frames.size()<<" extracted Frames from the left camera."<<std::endl;
    DAVISEvents events_r(std::string(AEDAT_FOLDER) + std::string(EVENTS_R), VERBOSE);
    std::cout<< events_r.polarity.size()<<" extracted Events from the left camera."<<std::endl;

    sync_frames(frames_l, frames_r);
    std::cout<<"Left frames number after sync: "<<frames_l.frames.size()<<"  Right frames number after sync: "<<frames_r.frames.size()<<std::endl;


    std::cout<<"Waiting your input for the frame position: ";
    std::cin>>framepos;


    printDavisStereo(framepos, frames_l, frames_r, events_l, events_r, t_halfspan, XDIM, YDIM, VERBOSE);


    return 0;
}

