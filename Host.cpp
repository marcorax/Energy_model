#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "../Kernels/Kernels.cl"
#define KERNEL_FUNC "convolute"

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

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}


/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

void save_results(std::string FileName, cv::Mat_<float> InputMat){
    
    cv::Mat tmp;
    cv::normalize( InputMat, tmp, 0, 65535, cv::NORM_MINMAX, CV_16UC1); 
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try {
        cv::imwrite(FileName+".png", tmp, compression_params);
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }

    fprintf(stdout, "Saved PNG file with alpha data.\n");
};

int main() {

   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int i, j, err;

   /* Data and buffers */
   float LGN_filter[LGN_RF_SIZE*LGN_RF_SIZE];
   cl_mem LGN_filter_buf, input_image_l, result_image_l_on, result_image_l_off,
   input_image_r, result_image_r_on, result_image_r_off,additional_data_buf;   
   int framepos;
   unsigned int t_halfspan = 1000;

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

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Query the device to read useful informations */
   size_t name_length;
   cl_uint compute_units;
   err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &name_length);  
   char device_name[name_length];
   err = clGetDeviceInfo(device, CL_DEVICE_NAME, name_length*sizeof(char), &device_name, &name_length); 
   if(err < 0) {
      perror("Couldn't find device name");
      exit(1);   
   }
   std::cout<<std::endl<<"Device name: "<<device_name<<std::endl;

   err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL); 
   if(err < 0) {
      perror("Couldn't find the number of compute devices");
      exit(1);   
   }
   std::cout<<std::endl<<"Number of compute devices: "<<compute_units<<std::endl;
   
   
   /* Build the program and create the kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);   
   }

   /* Create a command queue */
   queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   }; 
   
   /* Define the kernel space */
   size_t KernelXDIM = XDIM ;
   size_t KernelYDIM = YDIM + 4; //to have the global size as a multiple of 8
   size_t global_size[2], local_size[2]; // Process the entire image
   global_size[0] = KernelXDIM;
   global_size[1] = KernelYDIM; 
   //I have to find a way to set it up automatically
   local_size[0] = 8; 
   local_size[1] = 8;

   /* Create a buffer to hold the filter */
   LGN_filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
      CL_MEM_COPY_HOST_PTR, sizeof(LGN_filter), LGN_filter, &err);
   if(err < 0) {
      perror("Couldn't create the filter buffer object");
      exit(1);   
   }

   /* Create image buffers to hold the input pictures */ 
   // I don't know yet how to load multiple frames at once.
   float * DavisLeftFrame, * DavisRightFrame;
   DavisLeftFrame = (float*)frames_l.frames[framepos].data;
   DavisRightFrame = (float*)frames_r.frames[framepos].data;
   
   input_image_r = clCreateBuffer(context, CL_MEM_READ_ONLY | 
      CL_MEM_COPY_HOST_PTR, sizeof(float)*XDIM*YDIM, DavisRightFrame, &err);
   if(err < 0) {
      perror("Couldn't create the input buffer object");
      exit(1);   
   }

   /* Create image buffers to hold the results */
   
   result_image_r_on = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float)*XDIM*YDIM, NULL, &err);
   if(err < 0) {
      perror("Couldn't create the on result buffer object");
      exit(1);   
   }

   result_image_r_off = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float)*XDIM*YDIM, NULL, &err);
   if(err < 0) {
      perror("Couldn't create the on result buffer object");
      exit(1);   
   }
   
   /* Create a buffer to hold filter and image informations that should be 
   computed only once */
   int additional_data [4];
   additional_data [0] = XDIM;
   additional_data [1] = YDIM;
   additional_data [2] = LGN_RF_SIZE;
   additional_data [3] = LGN_RF_SIZE/2; //half filter size;

   additional_data_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
      CL_MEM_COPY_HOST_PTR, sizeof(additional_data), additional_data, &err);
   if(err < 0) {
      perror("Couldn't create the filter buffer object");
      exit(1);   
   }


   /* Set buffers as arguments to the kernel */
 
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image_r);
   if(err < 0) {
      perror("Couldn't set the left input image buffer as the kernel argument");
      exit(1);   
   }
 
   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_image_r_on);
   if(err < 0) {
      perror("Couldn't set the left result image buffer as the kernel argument");
      exit(1);   
   }

   err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_image_r_off);
   if(err < 0) {
      perror("Couldn't set the left result image buffer as the kernel argument");
      exit(1);   
   }


   err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &LGN_filter_buf);
   if(err < 0) {
      perror("Couldn't set the filter buffer as the kernel argument");
      exit(1);   
   }
   
   err = clSetKernelArg(kernel, 4, 
   sizeof(float)*(local_size[0]+additional_data[3])*(local_size[1]+additional_data[3]),
   NULL);
   if(err < 0) {
      perror("Couldn't set the local image memory as the kernel argument");
      exit(1);   
   }/*I'm allocating local memory to store all the pixel that will computed for convolution
   in each work group.
   for reference look at this page: https://www.evl.uic.edu/kreda/gpu/image-convolution/ */

   err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &additional_data_buf);
   if(err < 0) {
      perror("Couldn't set the additional data as the kernel argument");
      exit(1);   
   }
 
   size_t origin[] = {0,0,0};
   size_t region[] = {XDIM,YDIM,1};

   // Execute the OpenCL kernel on the list

   err = clEnqueueNDRangeKernel(queue, kernel, 2,
   NULL, global_size, local_size, 0, NULL, NULL);       
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }
   
    /* Enqueue command to read the results */
   float result_on[XDIM*YDIM], result_off[XDIM*YDIM]; //computation on a color image it will cause a segmentation fault
   err = clEnqueueReadBuffer(queue, result_image_r_on, CL_TRUE, 0,
    sizeof(float)*XDIM*YDIM, &result_on, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't read result left image from the buffer object");
      exit(1);   
   }

   err = clEnqueueReadBuffer(queue, result_image_r_off, CL_TRUE, 0,
    sizeof(float)*XDIM*YDIM, &result_off, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read result left image from the buffer object");
      exit(1);   
   }

   cv::Mat tmp_on(YDIM, XDIM, CV_32FC1, &result_on);
   cv::Mat tmp_off(YDIM, XDIM, CV_32FC1, &result_off);

   cv::namedWindow("Result ON of frames : " + std::to_string(framepos), cv::WINDOW_NORMAL);
   cv::resizeWindow("Result ON of frames : " + std::to_string(framepos), XDIM, YDIM);
   cv::imshow("Result ON of frames : " + std::to_string(framepos), tmp_on);
   
   cv::namedWindow("Result OFF of frames : " + std::to_string(framepos), cv::WINDOW_NORMAL);
   cv::resizeWindow("Result OFF of frames : " + std::to_string(framepos), XDIM, YDIM);
   cv::imshow("Result OFF of frames : " + std::to_string(framepos), tmp_off);


   /* Deallocate resources */
   clReleaseMemObject(LGN_filter_buf);
   clReleaseMemObject(input_image_r);
   clReleaseMemObject(result_image_r_on);
   clReleaseMemObject(result_image_r_off);
   clReleaseMemObject(additional_data_buf);
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

