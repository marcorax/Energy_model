//I still have to introduce NAN values for borders and to make it a real convolution by reversing the pixels
__kernel void convol_ker(
	__constant float * input, 
	__global float * pos_output,
	__global float * neg_output,
	__constant float * filter,
	__local float * cached,
    __constant int * additional_data)
{
    int image_w = get_global_size(0);
    int image_h = get_global_size(1);
    int filter_size = additional_data[0];
    int half_filter_size = additional_data[1];
    int twice_half_filter_size = additional_data[2];
    
	const int idx = get_global_id(0);
	const int idy = get_global_id(1);
	const int localRowLen = twice_half_filter_size + get_local_size(0);
	const int localRowOffset = ( get_local_id(1) + half_filter_size ) * localRowLen;
	const int myLocal = localRowOffset + get_local_id(0) + half_filter_size;		
		
	// I cache one pixel per worker in the local memory to speed up the convolution.
	pos_output[idx + idy*image_w] = 0;
	neg_output[idx + idy*image_w] = 0;
	cached[ myLocal ] = input[idx + idy*image_w];
	

	/*
	The pictures computed by this software have a resolution of 240x180
	Here I check if my current worker is looking at the borders of the image,
	if that is the case, the worker would have to stop computing, and leave a 1 as a result
	I need to be shure that the filter can fit the portion of the image controlled by the worker*/
	if (
		get_global_id(0) < half_filter_size 			|| 
		get_global_id(0) > image_w - half_filter_size - 1  	|| 
		get_global_id(1) < half_filter_size			|| 
		get_global_id(1) > image_h - half_filter_size - 1  )
	{
		// no computation for me, sync and exit
		pos_output[idx + idy*image_w] = 1;
		neg_output[idx + idy*image_w] = 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}
	else 
	{
		// copy additional elements
		int localColOffset = -1;
		int globalColOffset = -1;
		
		if ( get_local_id(0) < half_filter_size )
		{
			localColOffset = get_local_id(0);
			globalColOffset = -half_filter_size;
			
			cached[ localRowOffset + get_local_id(0) ] = input[idx - globalColOffset + idy*image_w];
		}
		if ( get_local_id(0) >= get_local_size(0) - half_filter_size )
		{
			localColOffset = get_local_id(0) + twice_half_filter_size;
			globalColOffset = half_filter_size;
			
			cached[ myLocal + half_filter_size ] = input[idx + globalColOffset + idy*image_w];
		}
		
		
		if ( get_local_id(1) < half_filter_size )
		{
			cached[ get_local_id(1) * localRowLen + get_local_id(0) + half_filter_size ] = input[idx + (idy-half_filter_size)*image_w];
			if (localColOffset >= 0)
			{
				cached[ get_local_id(1) * localRowLen + localColOffset ] = input[idx + globalColOffset + (idy-half_filter_size)*image_w];
			}
		}
		if ( get_local_id(1) >= get_local_size(1) -half_filter_size )
		{
			int localRowOffset = ( get_local_id(1) + twice_half_filter_size ) * localRowLen;
			cached[ localRowOffset + get_local_id(0) + half_filter_size ] = input[idx + (idy+half_filter_size)*image_w];
			if (localColOffset >= 0)
			{
				cached[ localRowOffset + localColOffset ] = input[idx + globalColOffset + (idy+half_filter_size)*image_w];
			}
		}
		
		// sync
		barrier(CLK_LOCAL_MEM_FENCE);

		
		// perform convolution
		int fIndex = 0;
		float sum = (float) 0.0;
		int curRow;
		for (int r = -half_filter_size; r <= half_filter_size; r++)
		{
			curRow = r * localRowLen;
			for (int c = -half_filter_size; c <= half_filter_size; c++, fIndex++)
			{	

				sum += (float) cached[ myLocal + curRow + c ] * filter[fIndex];
				
			}
		}
		
		if(sum>0){
			pos_output[idx + idy*image_w] = sum;}
		else if(sum<0){			
			neg_output[idx + idy*image_w] = -sum;}

			
	}			
}