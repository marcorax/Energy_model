#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "../Kernels/blank.cl"
#define KERNEL_FUNC "blank"

#define AEDAT_FOLDER  "../TestFiles/"
#define FRAMES_L "Frames_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define EVENTS_L "Events_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define FRAMES_R "Frames_R_Moving_Bar-2018_03_06_17_04_05.aedat"
#define EVENTS_R "Events_R_Moving_Bar-2018_03_06_17_04_05.aedat"

#define VERBOSE 1 //set to 0 di disable, 1 (or any other integer) to enable
#define XDIM 240
#define YDIM 180

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <string>

#include <Davisloading.hpp>
#include <opencv2/opencv.hpp>
#include <Dataplotting.hpp>

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
   float full_matrix[80], zero_matrix[80];
   const size_t buffer_origin[3] = {5*sizeof(float), 3, 0};
   const size_t host_origin[3] = {1*sizeof(float), 1, 0};
   const size_t region[3] = {4*sizeof(float), 4, 1};
   cl_mem matrix_buffer;
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
//   std::cout<<"Left frames number after sync: "<<frames_l.frames.size()<<"  Right frames number after sync: "<<frames_r.frames.size()<<std::endl;


//   std::cout<<"Waiting your input for the frame position: ";
//   std::cin>>framepos;
//   printDavisStereo(framepos, frames_l, frames_r, events_l, events_r, t_halfspan, XDIM, YDIM, VERBOSE);
   
   
   /* Initialize data */
   for(i=0; i<80; i++) {
      full_matrix[i] = i*1.0f;
      zero_matrix[i] = 0.0;
   }   
   
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
   };

   std::cout<<std::endl<<"Device name: "<<device_name<<std::endl;
   err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL); 
   if(err < 0) {
      perror("Couldn't find the number of compute devices:");
      exit(1);   
   };
   std::cout<<std::endl<<"Number of compute devices: "<<compute_units<<std::endl;

   /* Build the program and create the kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);   
   };

   /* Create a buffer to hold 80 floats */
   matrix_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | 
      CL_MEM_COPY_HOST_PTR, sizeof(full_matrix), full_matrix, &err);
   if(err < 0) {
      perror("Couldn't create a buffer object");
      exit(1);   
   }

   /* Set buffer as argument to the kernel */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix_buffer);
   if(err < 0) {
      perror("Couldn't set the buffer as the kernel argument");
      exit(1);   
   }

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };   
   
   /* Enqueue kernel */
   err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   /* Enqueue command to write to buffer */
   err = clEnqueueWriteBuffer(queue, matrix_buffer, CL_TRUE, 0,
         sizeof(full_matrix), full_matrix, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't write to the buffer object");
      exit(1);   
   }

   /* Enqueue command to read rectangle of data */
   err = clEnqueueReadBufferRect(queue, matrix_buffer, CL_TRUE, 
         buffer_origin, host_origin, region, 10*sizeof(float), 0, 
         10*sizeof(float), 0, zero_matrix, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the rectangle from the buffer object");
      exit(1);   
   }

   /* Display updated buffer */
   for(i=0; i<8; i++) {
      for(j=0; j<10; j++) {
         printf("%6.1f", zero_matrix[j+i*10]);
      }
      printf("\n");
   }

   /* Deallocate resources */
   clReleaseMemObject(matrix_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}

