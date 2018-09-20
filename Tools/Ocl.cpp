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

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
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
        std::cout<<"Couldn't create the program : error code "<<err<<" "<<getErrorString(err)<<std::endl;
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
    cl_kernel &kernel, cv::Mat_<float> &filter, cv::Mat_<float> &ResultON, cv::Mat_<float> &ResultOFF, int releaseMem, cl_mem (&buffers)[3]){
   
    /* OpenCL data structures */
    cl_int err;

    /* Data and buffers */
    cl_mem filter_buf, additional_data_buf,
    input_image=buffers[0],
    result_image_on=buffers[1],
    result_image_off=buffers[2];

    /*Arrays to momentarily store the results*/ 
    float ResultONArray[XDIM*YDIM], ResultOFFArray[XDIM*YDIM];

    /* Define the kernel space */
    size_t global_size[2], local_size[2]; // Process the entire image
    global_size[0] = XDIM;
    global_size[1] = YDIM; 

    //TODO find a way to crank these parameters automatically to get all the juice from both AMD and Nvidia current architectures.
    local_size[0] = 16; 
    local_size[1] = 18;

    float * FilterArray;
    FilterArray = (float*)filter.data;
    /* Create a buffer to hold the filter */
    filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(float)*pow(filter.cols,2), FilterArray, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the filter buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    /* Create a buffer to hold filter and image informations that should be 
    computed only once */
    int additional_data [3];
    additional_data [0] = filter.cols;
    additional_data [1] = filter.cols/2; // half filter size
    additional_data [2] = filter.cols -1; // filter contour (filtersize without its center)

    additional_data_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(additional_data), additional_data, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the additional data buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }


    /* Set buffers as arguments to the kernel */
 
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        std::cout<<"Couldn't set the input image buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
 
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_image_on);
    if(err < 0) {
        std::cout<<"Couldn't set the ON result image buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_image_off);
    if(err < 0) {
        std::cout<<"Couldn't set the OFF result image buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &filter_buf);
    if(err < 0) {
        std::cout<<"Couldn't set the filter buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
    /*I'm allocating local memory to store all the pixel that will computed for convolution
    in each work group.
    for reference look at this page: https://www.evl.uic.edu/kreda/gpu/image-convolution/ */
    err = clSetKernelArg(kernel, 4, 
    sizeof(float)*(local_size[0]+additional_data[2])*(local_size[1]+additional_data[2]),
    NULL);
    if(err < 0) {
        std::cout<<"Couldn't set the local image memory as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &additional_data_buf);
    if(err < 0) {
        std::cout<<"Couldn't set the additional data as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
    size_t origin[] = {0,0,0};
    size_t region[] = {XDIM,YDIM,1};

    // Execute the OpenCL kernel on the list

    err = clEnqueueNDRangeKernel(queue, kernel, 2,
    NULL, global_size, local_size, 0, NULL, NULL);       
    if(err < 0) {
        std::cout<<"Couldn't set the input image buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
        /* Enqueue command to read the results */
    err = clEnqueueReadBuffer(queue, result_image_on, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, ResultONArray, 0, NULL, NULL); 
    if(err < 0) {
        std::cout<<"Couldn't read the ON result image from the buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clEnqueueReadBuffer(queue, result_image_off, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, ResultOFFArray, 0, NULL, NULL);
    if(err < 0) {
        std::cout<<"Couldn't read the OFF result left image from the buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
    /* cv::Mat created by array are referenced to them, performing more than one 
    convolution in the same program would cause data corruption, for this reason
    I perform a deep copy of temporary matrices instead */
    cv::Mat tmpON(YDIM, XDIM, CV_32FC1, ResultONArray);
    cv::Mat tmpOFF(YDIM, XDIM, CV_32FC1, ResultOFFArray);

    ResultON = tmpON.clone();
    ResultOFF = tmpOFF.clone();

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
    cl_kernel &kernel, cv::Mat_<float> &filter, cv::Mat_<float> &Image,
    cv::Mat_<float> &ResultON, cv::Mat_<float> &ResultOFF,
    int allocateMem, int releaseMem, cl_mem (&buffers)[3]) {
   
    /* OpenCL data structures */
    cl_int err;

    /* Data and buffers */
    cl_mem filter_buf, input_image, result_image_on, result_image_off, additional_data_buf;  

    /*Arrays to momentarily store the results*/ 
    float ResultONArray[XDIM*YDIM], ResultOFFArray[XDIM*YDIM];

    if(allocateMem==CL_TRUE){
        /* Create a device and context */
        device = create_device();
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if(err < 0) {
            std::cout<<"Couldn't create a context : error code "<<err<<" "<<getErrorString(err)<<std::endl;
            exit(1);  
        }

        /* Query the device to read useful informations */
        size_t name_length;
        cl_uint compute_units;
        err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &name_length);  
        char device_name[name_length];
        err = clGetDeviceInfo(device, CL_DEVICE_NAME, name_length*sizeof(char), &device_name, &name_length); 
        if(err < 0) {
        std::cout<<"Couldn't find device name : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
        }
        std::cout<<std::endl<<"Device name: "<<device_name<<std::endl;  

        err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL); 
        if(err < 0) {
        std::cout<<"Couldn't find the number of compute devices : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
        }
        std::cout<<std::endl<<"Number of compute devices: "<<compute_units<<std::endl;
    
    
        /* Build the program and create the kernel */
        program = build_program(context, device, PROGRAM_FILE);
        kernel = clCreateKernel(program, CONVOL_KERNEL, &err);
        if(err < 0) {
            std::cout<<"Couldn't create a kernel : error code "<<err<<" "<<getErrorString(err)<<std::endl;
            exit(1);   
        }

        /* Create a command queue */
        queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
        if(err < 0) {
            std::cout<<"Couldn't create a command queue : error code "<<err<<" "<<getErrorString(err)<<std::endl;
            exit(1);   
        } 
    }
    /* Define the kernel space */
    size_t global_size[2], local_size[2]; // Process the entire image
    global_size[0] = XDIM;
    global_size[1] = YDIM; 
    //TODO find a way to crank these parameters automatically to get all the juice from both AMD and Nvidia current architectures.
    local_size[0] = 16; 
    local_size[1] = 18;
    
    float * FilterArray;
    FilterArray = (float*)filter.data;
    /* Create a buffer to hold the filter */
    filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(float)*pow(filter.cols,2), FilterArray, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the filter buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    float * ImageArray;
    ImageArray = (float*)Image.data;
    /* Create image buffers to hold the input pictures */ 
    input_image = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(float)*XDIM*YDIM, ImageArray, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the input buffer objec : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    /* Create image buffers to hold the results */
    result_image_on = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float)*XDIM*YDIM, NULL, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the ON result buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    result_image_off = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float)*XDIM*YDIM, NULL, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the OFF result buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
   
    /* Create a buffer to hold filter and image informations that should be 
    computed only once */
    int additional_data [3];
    additional_data [0] = filter.cols;
    additional_data [1] = filter.cols/2; // half filter size
    additional_data [2] = filter.cols -1; // filter contour (filtersize without its center)

    additional_data_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | 
        CL_MEM_COPY_HOST_PTR, sizeof(additional_data), additional_data, &err);
    if(err < 0) {
        std::cout<<"Couldn't create the filter buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }


    /* Set buffers as arguments to the kernel */
 
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        std::cout<<"Couldn't set the input image buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
 
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_image_on);
    if(err < 0) {
        std::cout<<"Couldn't set the ON result image buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_image_off);
    if(err < 0) {
        std::cout<<"Couldn't set the OFF result image buffer as the kernel argument "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }


    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &filter_buf);
    if(err < 0) {
        std::cout<<"Couldn't set the filter buffer as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
    /*I'm allocating local memory to store all the pixel that will computed for convolution
    in each work group.
    for reference look at this page: https://www.evl.uic.edu/kreda/gpu/image-convolution/ */
    err = clSetKernelArg(kernel, 4, 
    sizeof(float)*(local_size[0]+additional_data[2])*(local_size[1]+additional_data[2]),
    NULL);
    if(err < 0) {
        std::cout<<"Couldn't set the local image memory as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &additional_data_buf);
    if(err < 0) {
        std::cout<<"Couldn't set the additional data as the kernel argument : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
    size_t origin[] = {0,0,0};
    size_t region[] = {XDIM,YDIM,1};

    /* Execute the OpenCL kernel on the list */
    err = clEnqueueNDRangeKernel(queue, kernel, 2,
    NULL, global_size, local_size, 0, NULL, NULL);       
    if(err < 0) {
        std::cout<<"Couldn't enqueue the kernel : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }
    
    /* Enqueue command to read the results */
    err = clEnqueueReadBuffer(queue, result_image_on, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, ResultONArray, 0, NULL, NULL); 
    if(err < 0) {
        std::cout<<"Couldn't read the ON result left image from the buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    err = clEnqueueReadBuffer(queue, result_image_off, CL_TRUE, 0,
        sizeof(float)*XDIM*YDIM, ResultOFFArray, 0, NULL, NULL);
    if(err < 0) {
        std::cout<<"Couldn't read the OFF result left image from the buffer object : error code "<<err<<" "<<getErrorString(err)<<std::endl;
        exit(1);   
    }

    /* cv::Mat created by array are referenced to them, performing more than one 
    convolution in the same program would cause data corruption, for this reason
    I perform a deep copy of temporary matrices instead */
    cv::Mat tmpON(YDIM, XDIM, CV_32FC1, ResultONArray);
    cv::Mat tmpOFF(YDIM, XDIM, CV_32FC1, ResultOFFArray);

    ResultON = tmpON.clone();
    ResultOFF = tmpOFF.clone();
 
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
