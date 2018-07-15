#define TEST_FILE "Frames_L_Moving_Bar-2018_03_06_17_04_05.aedat"
#define XDIM 240
#define YDIM 180

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <pngwriter.h>
#include <fstream>
#include <string>
#include <iostream>
#include <Davisloading.hpp>

int main(){
    DAVISFrames testobj(TEST_FILE, XDIM, YDIM);
/*     pngwriter png(XDIM,YDIM,0,"test.png");
    int xcursor=0;
    int ycursor=0;
    for(unsigned int i=0; i<XDIM*YDIM; i++){
        png.plot(xcursor,ycursor,testobj.frames[0][i],testobj.frames[0][i],testobj.frames[0][i]);
        if(xcursor==XDIM-1){//I reached the end of the line, I move to the next one
            xcursor=0;
            ycursor++;
        }
        xcursor++;
    }
    png.close(); */
    return 0;
}

