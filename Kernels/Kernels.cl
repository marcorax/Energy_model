__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void convolute(
	__read_only image2d_t input, 
	__write_only image2d_t output,
	__constant float * filter,
	__local float4 * cached,
    __constant unsigned int * additional_data)
{
    unsigned int image_w = additional_data[0];
    unsigned int image_h = additional_data[1];
    unsigned int filter_size = additional_data[2];
    unsigned int half_filter_size = additional_data[3];
    unsigned int twice_half_filter_size = half_filter_size*2;
    
	const int idy = get_global_id(1);
	const int idx = get_global_id(0);
	
	const int localRowLen = twice_half_filter_size + get_local_size(0);
	const int localRowOffset = ( get_local_id(1) + half_filter_size ) * localRowLen;
	const int myLocal = localRowOffset + get_local_id(0) + half_filter_size;		
		
	// copy my pixel
	cached[ myLocal ] = read_imagef(input, image_sampler, (int2) (idx, idy));

	/*the -4 below is used to avoid access to non existent memory
	The pictures computed by this software have a resolution of 240x180
	The work group size is set to be 8x8, hence the total kernel dimension is 
	240x184
	I might apply offsets to build a more general convolution kernel 
	but it might worsen the readability*/
	if (
		get_global_id(0) < half_filter_size 			|| 
		get_global_id(0) > image_w - half_filter_size - 1  	|| 
		get_global_id(1) < half_filter_size			|| 
		get_global_id(1) > image_h - half_filter_size - 1  -4 
	)
	{
		// no computation for me, sync and exit
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
			
			cached[ localRowOffset + get_local_id(0) ] = read_imagef(input, image_sampler, (int2) (idx-half_filter_size, idy));
		}
		else if ( get_local_id(0) >= get_local_size(0) - half_filter_size )
		{
			localColOffset = get_local_id(0) + twice_half_filter_size;
			globalColOffset = half_filter_size;
			
			cached[ myLocal + half_filter_size ] = read_imagef(input, image_sampler, (int2) (idx+half_filter_size, idy));
		}
		
		
		if ( get_local_id(1) < half_filter_size )
		{
			cached[ get_local_id(1) * localRowLen + get_local_id(0) + half_filter_size ] = read_imagef(input, image_sampler, (int2) (idx,idy-half_filter_size));
			if (localColOffset > 0)
			{
				cached[ get_local_id(1) * localRowLen + localColOffset ] = read_imagef(input, image_sampler, (int2) (idx + globalColOffset, idy-half_filter_size));
			}
		}
		else if ( get_local_id(1) >= get_local_size(1) -half_filter_size )
		{
			int offset = ( get_local_id(1) + twice_half_filter_size ) * localRowLen;
			cached[ offset + get_local_id(0) + half_filter_size ] = read_imagef(input, image_sampler, (int2) (idx,idy+half_filter_size));
			if (localColOffset > 0)
			{
				cached[ offset + localColOffset ] = read_imagef(input, image_sampler, (int2) (idx + globalColOffset, idy+half_filter_size));
			}
		}
		
		// sync
		barrier(CLK_LOCAL_MEM_FENCE);

		
		// perform convolution
		int fIndex = 0;
		float4 sum = (float4) 0.0;
		
		for (int r = -half_filter_size; r <= half_filter_size; r++)
		{
			int curRow = r * localRowLen;
			for (int c = -half_filter_size; c <= half_filter_size; c++, fIndex++)
			{	
				sum += cached[ myLocal + curRow + c ] * filter[ fIndex ]; 
			}
		}
		write_imagef(output, (int2)(idx,idy), sum);

	}
}