#define PROGRAM_FILE "../Kernels/Kernels.cl"
#define CONVOL_KERNEL "convol_ker"
#define XDIM 240  //Davis horizontal resolution
#define YDIM 180  //Davis vertical resolution

#include <opencv2/opencv.hpp>
#include <cmath>

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
}

/* convolution in my labraries has twos different versions, depending if you want to continue to apply different convolutions on the same couple of images 
without rewriting the buffers*/ 


void convolution(cl_device_id &device, cl_context &context, cl_command_queue &queue, cl_program &program,
    cl_kernel &kernel, float * filter, int filtersize, float (&ResultON)[XDIM*YDIM], float (&ResultOFF)[XDIM*YDIM], int releaseMem, cl_mem (&buffers)[3]){
   
    /* OpenCL data structures */
    cl_int err;

    /* Data and buffers */
    cl_mem filter_buf, additional_data_buf,
    input_image=buffers[0],
    result_image_on=buffers[1],
    result_image_off=buffers[2];
 
    /* Define the kernel space */
    size_t global_size[2], local_size[2]; // Process the entire image
    global_size[0] = XDIM;
    global_size[1] = YDIM; 
    //I have to find a way to set it up automatically
    local_size[0] = 8; 
    local_size[1] = 8;

    /* Create a buffer to hold the filter */
    filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(filter)*pow(filtersize,2), filter, &err);
    if(err < 0) {
        perror("Couldn't create the filter buffer object");
        exit(1);   
    }

    /* Create a buffer to hold filter and image informations that should be 
    computed only once */
    int additional_data [3];
    additional_data [0] = filtersize;
    additional_data [1] = filtersize/2; // half filter size
    additional_data [2] = filtersize -1; // filter contour (filtersize without its center)

    additional_data_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(additional_data), additional_data, &err);
    if(err < 0) {
        perror("Couldn't create the filter buffer object");
        exit(1);   
    }


    /* Set buffers as arguments to the kernel */
 
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("Couldn't set the input image buffer as the kernel argument");
        exit(1);   
    }
 
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_image_on);
    if(err < 0) {
        perror("Couldn't set the ON result image buffer as the kernel argument");
        exit(1);   
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_image_off);
    if(err < 0) {
        perror("Couldn't set the OFF result image buffer as the kernel argument");
        exit(1);   
    }


    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &filter_buf);
    if(err < 0) {
        perror("Couldn't set the filter buffer as the kernel argument");
        exit(1);   
    }
    
    err = clSetKernelArg(kernel, 4, 
    sizeof(float)*(local_size[0]+additional_data[1])*(local_size[1]+additional_data[1]),
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
    err = clEnqueueReadBuffer(queue, result_image_on, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, &ResultON, 0, NULL, NULL); 
    if(err < 0) {
        perror("Couldn't read the ON result image from the buffer object");
        exit(1);   
    }

    err = clEnqueueReadBuffer(queue, result_image_off, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, &ResultOFF, 0, NULL, NULL);
    if(err < 0) {
        perror("Couldn't read the OFF result left image from the buffer object");
        exit(1);   
    }

    if(releaseMem==CL_TRUE){
        clReleaseMemObject(input_image);
        clReleaseMemObject(result_image_on);
        clReleaseMemObject(result_image_off);
        buffers[0]=nullptr;
        buffers[1]=nullptr;
        buffers[2]=nullptr;
    }
    else{
        buffers[0]=input_image;
        buffers[1]=result_image_on;
        buffers[2]=result_image_off;
    }

    clReleaseMemObject(filter_buf);
    clReleaseMemObject(additional_data_buf);
}

void convolution(cl_device_id &device, cl_context &context, cl_command_queue &queue, cl_program &program,
    cl_kernel &kernel, float * filter, int filtersize, cv::Mat_<float> &Image,
    float (&ResultON)[XDIM*YDIM], float (&ResultOFF)[XDIM*YDIM],
    int allocateMem, int releaseMem, cl_mem (&buffers)[3]) {
   
    /* OpenCL data structures */
    cl_int err;

    /* Data and buffers */
    cl_mem filter_buf, input_image, result_image_on, result_image_off, additional_data_buf;   
   
    if(allocateMem==CL_TRUE){
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
        kernel = clCreateKernel(program, CONVOL_KERNEL, &err);
        if(err < 0) {
            perror("Couldn't create a kernel");
            exit(1);   
        }

        /* Create a command queue */
        queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
        if(err < 0) {
            perror("Couldn't create a command queue");
            exit(1);   
        } 
    }
    /* Define the kernel space */
    size_t global_size[2], local_size[2]; // Process the entire image
    global_size[0] = XDIM;
    global_size[1] = YDIM; 
    //I have to find a way to set it up automatically
    local_size[0] = 8; 
    local_size[1] = 8;

    /* Create a buffer to hold the filter */
    filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(filter)*pow(filtersize,2), filter, &err);
    if(err < 0) {
        perror("Couldn't create the filter buffer object");
        exit(1);   
    }

    /* Create image buffers to hold the input pictures */ 
    // I don't know yet how to load multiple frames at once.
    float * ImageArray;
    ImageArray = (float*)Image.data;
   
    input_image = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(float)*XDIM*YDIM, ImageArray, &err);
    if(err < 0) {
        perror("Couldn't create the input buffer object");
        exit(1);   
    }

    /* Create image buffers to hold the results */
    result_image_on = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float)*XDIM*YDIM, NULL, &err);
    if(err < 0) {
        perror("Couldn't create the ON result buffer object");
        exit(1);   
    }

    result_image_off = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float)*XDIM*YDIM, NULL, &err);
    if(err < 0) {
        perror("Couldn't create the OFF result buffer object");
        exit(1);   
    }
   
    /* Create a buffer to hold filter and image informations that should be 
    computed only once */
    int additional_data [3];
    additional_data [0] = filtersize;
    additional_data [1] = filtersize/2; // half filter size
    additional_data [2] = filtersize -1; // filter contour (filtersize without its center)

    additional_data_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(additional_data), additional_data, &err);
    if(err < 0) {
        perror("Couldn't create the filter buffer object");
        exit(1);   
    }


    /* Set buffers as arguments to the kernel */
 
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("Couldn't set the input image buffer as the kernel argument");
        exit(1);   
    }
 
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_image_on);
    if(err < 0) {
        perror("Couldn't set the ON result image buffer as the kernel argument");
        exit(1);   
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_image_off);
    if(err < 0) {
        perror("Couldn't set the OFF result image buffer as the kernel argument");
        exit(1);   
    }


    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &filter_buf);
    if(err < 0) {
        perror("Couldn't set the filter buffer as the kernel argument");
        exit(1);   
    }
    
    err = clSetKernelArg(kernel, 4, 
    sizeof(float)*(local_size[0]+additional_data[1])*(local_size[1]+additional_data[1]),
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
    err = clEnqueueReadBuffer(queue, result_image_on, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, &ResultON, 0, NULL, NULL); 
    if(err < 0) {
        perror("Couldn't read the ON result left image from the buffer object");
        exit(1);   
    }

    err = clEnqueueReadBuffer(queue, result_image_off, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, &ResultOFF, 0, NULL, NULL);
    if(err < 0) {
        perror("Couldn't read the OFF result left image from the buffer object");
        exit(1);   
    }
 
    if(releaseMem==CL_TRUE){
        clReleaseMemObject(input_image);
        clReleaseMemObject(result_image_on);
        clReleaseMemObject(result_image_off);
        buffers[0]=nullptr;
        buffers[1]=nullptr;
        buffers[2]=nullptr;
    }
    else{
        buffers[0]=input_image;
        buffers[1]=result_image_on;
        buffers[2]=result_image_off;
    }

    clReleaseMemObject(filter_buf);
    clReleaseMemObject(additional_data_buf);

}
