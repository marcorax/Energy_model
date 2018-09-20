#ifndef DATAPLOTTING_HPP
#define DATAPLOTTING_HPP
#include <Davisloading.hpp>

//TODO add a description to functions and variables
void printDavisStereo(unsigned int, DAVISFrames &, DAVISFrames &, DAVISEvents &, DAVISEvents &,
                      unsigned int, int ,int ,const int & );

void printOnOffImages(std::string Windowmsg, cv::Mat_<float> on_image, cv::Mat_<float> off_image, unsigned int magnifingFactor);

#endif