#ifndef DAVISLOADING_HPP
#define DAVISLOADING_HPP
#include <vector>
#include <opencv2/opencv.hpp>


class DAVISFrames {
    /* Obj loading frames from AEDAT 3.1 Davis sensors */
    //std::list frames;
    public:
    DAVISFrames(std::string, int , int , const int &);
    std::vector <cv::Mat> frames;
    std::vector <unsigned int> start_ts;
    std::vector <unsigned int> end_ts;
    
    private:
    
    std::string filename;
    int xdim, ydim;     /* xdim and ydim are the camera dimension in pixel */

};

class DAVISEvents {
    /* Obj loading frames from AEDAT 3.1 Davis sensors */
    //std::list frames;
    public:
    DAVISEvents(std::string, const int &);
    std::vector <unsigned short> polarity;
    std::vector <unsigned int> timestamp;
    std::vector <unsigned int> x_addr;
    std::vector <unsigned int> y_addr;
    
    private:
    
    std::string filename;

};

void sync_frames(DAVISFrames & , DAVISFrames & );

#endif