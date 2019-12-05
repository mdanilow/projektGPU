#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16


// __device__ void clearBorders()


__global__ void localCCL(const double* input, double* output, const int height, const int width){

    __shared__ int segments[BLOCK_WIDTH * BLOCK_HEIGHT];
    __shared__ int labels[BLOCK_WIDTH * BLOCK_HEIGHT];
    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int localRow = threadIdx.x;
    int localCol = threadIdx.y;
    int localIndex = localCol * blockDim.y + localRow;
    int globalIndex = col * height + row;

    // load corresponding image tile to shared memory
    segments[localIndex] = input[globalIndex];

    // clear borders in every tile
    if(threadIdx.x == 0 || threadIdx.x == BLOCK_WIDTH-1 || threadIdx.y == 0 || threadIdx.y == BLOCK_HEIGHT-1){
        segments[localIndex] = 0;
    }

    if(col < width && row < height)
        output[globalIndex] = segments[localIndex];
}