#define TEST_FILE "Test Files/Frames_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define XDIM 240
#define YDIM 180

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <Davisloading.hpp>

int main(){
    DAVISFrames testobj(TEST_FILE, XDIM, YDIM);
    std::cout<< testobj.frames.size();
    return 0;
}

