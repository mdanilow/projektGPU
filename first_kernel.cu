#define TILE_W 16
#define TILE_H 16
#define R 	1
#define D 	(2*R+1)
#define S 	(D*D)
#define BLOCK_W (TILE_W+2*R)
#define BLOCK_H (TILE_H+2*R)


__global__ void convFilter(const double* input, const double* coefs, 
			double* output,	const int width, const int height)
{
//    __shared__ double sharedmem[BLOCK_H*BLOCK_W];
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(x >= R && x <= width-R && y >= R && y <= height-R)
    {
        int index = x + y * width;
        for(int i=-R; i<=R; i++)
        {
            for(int j=-R; j<=R; j++)
            {
                int pixelpos = (x+j) + (y+i)*width;
                int coefpos = (j+R) + D*(i+R);
                output[index] += input[pixelpos] * coefs[coefpos];
            }
        }
    }
}

