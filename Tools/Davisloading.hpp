#ifndef DAVISLOADING_HPP
#define DAVISLOADING_HPP
#include <vector>
#include <opencv2/opencv.hpp>


class DAVISFrames {
    /* Obj loading frames from AEDAT 3.1 Davis sensors */
    //std::list frames;
    public:
    DAVISFrames(std::string fn, int dimx, int dimy);
    std::vector <cv::Mat> frames;
    std::vector <unsigned int> start_ts;
    std::vector <unsigned int> end_ts;
    
    private:
    
    std::string filename;
    int xdim, ydim;     /* xdim and ydim are the camera dimension in pixel */

};

#endif