#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "../Kernels/Kernels.cl"
#define KERNEL_FUNC "convol_ker"

#define AEDAT_FOLDER  "../TestFiles/"
#define FRAMES_L "Frames_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define EVENTS_L "Events_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define FRAMES_R "Frames_R_Moving_Bar-2018_03_06_17_04_05.aedat"
#define EVENTS_R "Events_R_Moving_Bar-2018_03_06_17_04_05.aedat"

#define VERBOSE 1 //set to 0 di disable, 1 (or any other integer) to enable
#define XDIM 240  //Davis horizontal resolution
#define YDIM 180  //Davis vertical resolution
#define LGN_RF_SIZE 3
#define FRAMEPOS 50

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <string>

#include <Davisloading.hpp>
#include <opencv2/opencv.hpp>
#include <Dataplotting.hpp>
#include <Filters.hpp>
#include <Ocl.hpp>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


int main() {

    /* OpenCL data structures */
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    /* Data and buffers */
    int framepos;
    unsigned int numPhases = 8;
    unsigned int numOrientations = 8;
    unsigned int t_halfspan = 1000;

    /* Array containing all filters for edge detection and phase coding */
    //TODO use the same equation and parameters in the Parvo Path too
    int gaborSize = 3;
    double sigma = 3;
    double freq = 4;
    double phase, theta;

    /* It cycles through all orientations and then it changes phase */
    cv::Mat_<float> Filters[numPhases*numOrientations];
    for(unsigned int i=0; i<numPhases; i++){
        for(unsigned int j=0; j<numOrientations; j++){
            phase = (double)-90 + 180/numPhases * i;
            theta = (double)180/numPhases * j;
            Filters[j+i*numPhases] = gaborFilter(gaborSize, sigma, theta, freq, phase);
        }
    }
   
    /* How to check if the implemented filters are what we want */
    printOnOffImages("First Filter", Filters[0], -Filters[0], 50);
    printOnOffImages("Fourth Filter", Filters[4], -Filters[4], 50);
    printOnOffImages("Seventh Filter", Filters[7], -Filters[7], 50);


    /* Buffers carrying result and input image for the convolutions */
    cl_mem conv_buffers[3];
        
    /*  Array to temporary store the results */
    float result_on[XDIM*YDIM], result_off[XDIM*YDIM];

    /* Array of Mat holding the oriented bordered of the image */
    cv::Mat_<float> BoundaryResults_on[numOrientations];
    cv::Mat_<float> BoundaryResults_off[numOrientations];

    
    /* Loading Davis Data */
    DAVISFrames frames_l(std::string(AEDAT_FOLDER) + std::string(FRAMES_L), XDIM, YDIM, VERBOSE);
    std::cout<< frames_l.frames.size()<<" extracted Frames from the left camera."<<std::endl;
    DAVISEvents events_l(std::string(AEDAT_FOLDER) + std::string(EVENTS_L), VERBOSE);
    std::cout<< events_l.polarity.size()<<" extracted Events from the left camera."<<std::endl;
    DAVISFrames frames_r(std::string(AEDAT_FOLDER) + std::string(FRAMES_R), XDIM, YDIM, VERBOSE);
    std::cout<< frames_r.frames.size()<<" extracted Frames from the left camera."<<std::endl;
    DAVISEvents events_r(std::string(AEDAT_FOLDER) + std::string(EVENTS_R), VERBOSE);
    std::cout<< events_r.polarity.size()<<" extracted Events from the left camera."<<std::endl;
   
    /* Davis cameras at the time of writing cannot sync frame capure and exposure time
    for this reason i need to sync the recordings selecting only the more synchronised couple of left and right frames. */    
    sync_frames(frames_l, frames_r);
    std::cout<<"Left frames number after sync: "<<frames_l.frames.size()<<"  Right frames number after sync: "<<frames_r.frames.size()<<std::endl;

    framepos = FRAMEPOS;
    //std::cout<<"Waiting your input for the frame position: ";
    //std::cin>>framepos;
    printDavisStereo(framepos, frames_l, frames_r, events_l, events_r, t_halfspan, XDIM, YDIM, VERBOSE);


    /*                                                   */
    /* Simple BCS (Boundary Countour System) computation */
    /*                                                   */

    convolution(device, context, queue, program, kernel, Filters[0], frames_r.frames[framepos],
        BoundaryResults_on[0], BoundaryResults_off[0], CL_TRUE, CL_FALSE, conv_buffers);

    /* This overrided convolution is used to avoid to build again the buffers */
    for(unsigned int i = 1; i<numOrientations-1; i++){
    convolution(device, context, queue, program, kernel, Filters[i], 
        BoundaryResults_on[i], BoundaryResults_off[i], CL_FALSE, conv_buffers);
    }
    
    /* The last concolution have the parameter to deallocate memory set to CL_TRUE */
    convolution(device, context, queue, program, kernel, Filters[numOrientations-1], 
        BoundaryResults_on[numOrientations-1], BoundaryResults_off[numOrientations-1], CL_TRUE, conv_buffers);


    cv::namedWindow("Result ON of frames : " + std::to_string(framepos), cv::WINDOW_NORMAL);
    cv::resizeWindow("Result ON of frames : " + std::to_string(framepos), XDIM, YDIM);
    cv::imshow("Result ON of frames : " + std::to_string(framepos), BoundaryResults_on[0]);
    
    cv::namedWindow("Result OFF of frames : " + std::to_string(framepos), cv::WINDOW_NORMAL);
    cv::resizeWindow("Result OFF of frames : " + std::to_string(framepos), XDIM, YDIM);
    cv::imshow("Result OFF of frames : " + std::to_string(framepos), BoundaryResults_off[0]);

    
    printOnOffImages("Mixed result", BoundaryResults_on[0], BoundaryResults_off[0], 1);

    /* Deallocate resources */

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    /* Check pixels  
    std::cout<<std::endl<<tmp_off.at<float>(118,7)<<" "<<tmp_off.at<float>(118,8)<<" "<<tmp_off.at<float>(118,9)<<" "<<std::endl;
    std::cout<<std::endl<<tmp_off.at<float>(119,7)<<" "<<tmp_off.at<float>(119,8)<<" "<<tmp_off.at<float>(119,9)<<" "<<std::endl;
    std::cout<<std::endl<<tmp_off.at<float>(120,7)<<" "<<tmp_off.at<float>(120,8)<<" "<<tmp_off.at<float>(120,9)<<" "<<std::endl;
    */

    /* Save Results
    save_results("../Result_r_OFF.png", tmp_off);
    */

    cv::waitKey(0); 
    
    return 0;
}

