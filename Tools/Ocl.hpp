

#include <opencv2/opencv.hpp>


#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device();

/* Create program from a file and compile it */
cl_program build_program(cl_context, cl_device_id, const char*);

/* It saves results from the kernels (cv::Mat of floats) */
void save_results(std::string, cv::Mat_<float>);

void convolution(cl_device_id &device, cl_context &context, cl_command_queue &queue, cl_program &program,
    cl_kernel &kernel, cv::Mat_<float> &filter, cv::Mat_<float> &ResultON, cv::Mat_<float> &ResultOFF,
     int releaseMem, cl_mem (&buffers)[3]);

void convolution(cl_device_id &device, cl_context &context, cl_command_queue &queue, cl_program &program,
    cl_kernel &kernel, cv::Mat_<float> &filter, cv::Mat_<float> &Image,
    cv::Mat_<float> &ResultON, cv::Mat_<float> &ResultOFF,
    int allocateMem, int releaseMem, cl_mem (&buffers)[3]); 