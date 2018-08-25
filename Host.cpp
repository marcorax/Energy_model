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

int main() {

   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int i, j, err;

   /* Data and buffers */
   float LGN_filter[LGN_RF_SIZE][LGN_RF_SIZE];
   cl_mem LGN_filter_buf, input_image_l, result_image_l,
   additional_data_buf;   
   int framepos;
   unsigned int t_halfspan = 1000;
   struct _cl_image_format DavisFormat, ResultFormat, tmpFormat;
   struct _cl_image_desc DavisDesc, ResultDesc, tmpDesc;

   DavisFormat.image_channel_data_type = CL_FLOAT;
   DavisFormat.image_channel_order = CL_R;

   DavisDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
   DavisDesc.image_width = XDIM;
   DavisDesc.image_height = YDIM;
   DavisDesc.image_depth = 1;
   DavisDesc.image_array_size = 1;
   DavisDesc.image_row_pitch = 0;
   DavisDesc.image_slice_pitch = 0;
   DavisDesc.num_mip_levels = 0;
   DavisDesc.num_samples = 0;
   DavisDesc.buffer = NULL;

   ResultFormat = DavisFormat;
   ResultDesc = DavisDesc;
  

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
   
   /* Define the kernel space */
   int KernelXDIM = XDIM ;
   int KernelYDIM = YDIM + 4; //to have the global size as a multiple of 8
   size_t global_size[2]; // Process the entire image
   global_size[0] = KernelXDIM;
   global_size[1] = KernelYDIM; 
   size_t local_size [2]; //I have to find a way to set it up automatically
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
   float* DavisData[2];
   DavisData[0] = (float*)frames_l.frames[framepos].data;
   DavisData[1] = (float*)frames_r.frames[framepos].data;
   
   input_image_l = clCreateImage(context, CL_MEM_READ_ONLY | 
      CL_MEM_COPY_HOST_PTR, &DavisFormat, &DavisDesc, DavisData[0], &err);
   if(err < 0) {
      perror("Couldn't create the left input image buffer object");
      exit(1);   
   }

   /* Create image buffers to hold the results */
   result_image_l = clCreateImage(context, CL_MEM_WRITE_ONLY | 
      CL_MEM_ALLOC_HOST_PTR, &ResultFormat, &ResultDesc, NULL, &err);
   if(err < 0) {
      perror("Couldn't create the result image buffer object");
      exit(1);   
   }

   /* Create a buffer to hold filter and image informations that should be 
   computed only once */
   unsigned int additional_data [4];
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
 
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image_l);
   if(err < 0) {
      perror("Couldn't set the left input image buffer as the kernel argument");
      exit(1);   
   }

   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_image_l);
   if(err < 0) {
      perror("Couldn't set the left result image buffer as the kernel argument");
      exit(1);   
   }

   err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &LGN_filter_buf);
   if(err < 0) {
      perror("Couldn't set the filter buffer as the kernel argument");
      exit(1);   
   }
   
   err = clSetKernelArg(kernel, 3, 
   sizeof(cl_float4)*(local_size[0]+additional_data[4])*(local_size[1]+additional_data[4]),
   NULL);
   if(err < 0) {
      perror("Couldn't set the local image memory as the kernel argument");
      exit(1);   
   }/*I'm allocating local memory to store all the pixel that will computed for convolution
   in each work group.
   for reference look at this page: https://www.evl.uic.edu/kreda/gpu/image-convolution/ */

   err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &additional_data_buf);
   if(err < 0) {
      perror("Couldn't set the additional data as the kernel argument");
      exit(1);   
   }

   /* Create a command queue */
   queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };   

   // Execute the OpenCL kernel on the list

   err = clEnqueueNDRangeKernel(queue, kernel, 2,
   NULL, global_size, local_size, 0, NULL, NULL);       
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   size_t origin[] = {0,0,0};
   size_t region[] = {XDIM,YDIM};

   /* Enqueue command to write to the filter buffer 
   err = clEnqueueWriteBuffer(queue, LGN_filter_buf, CL_TRUE, 0,
         sizeof(LGN_filter), LGN_filter, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't write the filter to the buffer object");
      exit(1);   
   }

   /* Enqueue command to write to the input image buffers 
   err = clEnqueueWriteImage(queue, input_image_l, CL_TRUE, origin,
         region, XDIM*sizeof(float), XDIM*YDIM*sizeof(float), DavisData[0],
         0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't write input left image to the buffer object");
      exit(1);   
   }
   err = clEnqueueWriteImage(queue, input_image_r, CL_TRUE, origin,
         region, XDIM*sizeof(float), XDIM*YDIM*sizeof(float), DavisData[1],
         0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't write input right image to the buffer object");
      exit(1);   
   }

   /* Enqueue command to read the results */
   float * result[2];
   err = clEnqueueReadImage(queue, result_image_l, CL_TRUE, origin,
         region, XDIM*sizeof(float), XDIM*YDIM*sizeof(float), result[0],
         0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't read result left image from the buffer object");
      exit(1);   
   }
   
   /* Display updated buffer 
   for(i=0; i<8; i++) {
      for(j=0; j<10; j++) {
         printf("%6.1f", zero_matrix[j+i*10]);
      }
      printf("\n");
   }
*/

   /* Deallocate resources */
   clReleaseMemObject(LGN_filter_buf);
   clReleaseMemObject(input_image_l);
   clReleaseMemObject(result_image_l);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}

