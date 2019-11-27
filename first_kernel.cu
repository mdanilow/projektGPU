#define TILE_W 16
#define TILE_H 16
#define R 	1
#define D 	(2*R+1)
#define S 	(D*D)
#define BLOCK_W (TILE_W+2*R)
#define BLOCK_H (TILE_H+2*R)


__global__ void convFilter(const double* input, const double* coefs, 
			double* output,	const int width, const int height){
//    __shared__ double sharedmem[BLOCK_H*BLOCK_W];
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int index = row + col * height;
    
    if(row == 69 || col == 420)
        output[index] = 255;
    else
        output[index] = 0;
}

