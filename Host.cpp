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
   float LGN_filter[LGN_RF_SIZE*LGN_RF_SIZE];
   int framepos;
   unsigned int t_halfspan = 1000;
   cl_mem multipleconv_buffers[3];
   
   float result_l_on[XDIM*YDIM], result_l_off[XDIM*YDIM],
   result_r_on[XDIM*YDIM], result_r_off[XDIM*YDIM]; //computation on a color image it will cause a segmentation fault

   LGN(LGN_filter, LGN_RF_SIZE);

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

   framepos = FRAMEPOS;
   //std::cout<<"Waiting your input for the frame position: ";
   //std::cin>>framepos;
   printDavisStereo(framepos, frames_l, frames_r, events_l, events_r, t_halfspan, XDIM, YDIM, VERBOSE);

   convolution(device, context, queue, program, kernel, LGN_filter, LGN_RF_SIZE, frames_l.frames[framepos],
    result_l_on, result_l_off, CL_TRUE, CL_FALSE, multipleconv_buffers);

   convolution(device, context, queue, program, kernel, LGN_filter, LGN_RF_SIZE, 
    result_l_on, result_l_off, CL_TRUE, multipleconv_buffers);

   cv::Mat tmp_on(YDIM, XDIM, CV_32FC1, &result_l_on);
   cv::Mat tmp_off(YDIM, XDIM, CV_32FC1, &result_l_off);

   cv::namedWindow("Result ON of frames : " + std::to_string(framepos), cv::WINDOW_NORMAL);
   cv::resizeWindow("Result ON of frames : " + std::to_string(framepos), XDIM, YDIM);
   cv::imshow("Result ON of frames : " + std::to_string(framepos), tmp_on);
   
   cv::namedWindow("Result OFF of frames : " + std::to_string(framepos), cv::WINDOW_NORMAL);
   cv::resizeWindow("Result OFF of frames : " + std::to_string(framepos), XDIM, YDIM);
   cv::imshow("Result OFF of frames : " + std::to_string(framepos), tmp_off);


   /* Deallocate resources */

   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
  
   std::cout<<std::endl<<tmp_off.at<float>(118,7)<<" "<<tmp_off.at<float>(118,8)<<" "<<tmp_off.at<float>(118,9)<<" "<<std::endl;
   std::cout<<std::endl<<tmp_off.at<float>(119,7)<<" "<<tmp_off.at<float>(119,8)<<" "<<tmp_off.at<float>(119,9)<<" "<<std::endl;
   std::cout<<std::endl<<tmp_off.at<float>(120,7)<<" "<<tmp_off.at<float>(120,8)<<" "<<tmp_off.at<float>(120,9)<<" "<<std::endl;


   save_results("../Result_r_OFF.png", tmp_off);

   cv::waitKey(0); 
   
   return 0;
}

