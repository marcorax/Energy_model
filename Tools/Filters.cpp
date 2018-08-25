#include <Filters.hpp>
//each filter is considered to be a square filter with linear odd dimension of lsize (3x3 5x5 7x7 and so on)
//each function needs to receive a preallocated filter 2D array with lszie linear dimensions


void LGN(unsigned int & lsize, float ** filter)
{
    float normal_factor = lsize^2 - 1;
    for(int i = 0; i<lsize; i++){
        for(int j = 0; j<lsize; j++){
            filter[i][j] = -1 / normal_factor;            
        }
    }

    filter[lsize/2][lsize/2] = 1;
}
