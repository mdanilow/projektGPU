#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

__device__ bool inLeftBorder();
__device__ bool inRightBorder();
__device__ bool inTopBorder();
__device__ bool inBottomBorder();
__device__ int globalAddr(const int x, const int y, const int height);
__device__ int findRoot(int equivalenceArray[], int elementAddress);
__device__ void Union(int equivalenceArray[], const int elementAddress0, const int elementAddress1);


__global__ void mergeTiles(
        const double* dSegData,
        int* dLabelsData,
        const int height,
        const int width){

    __shared__ int changed;

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int depth = blockDim.z;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int localDepth = threadIdx.z;

    int subBlockDim = 10;
    int x, y = 0;

    int repetitions = int(subBlockDim/depth); //how many times are the thread reused for the given subblock?
    //shared sChanged[1]; //shared memory used to check whether the solution is final or not

    while(1) {
        if(localRow == 0 && localCol == 0 && localDepth == 0) changed = 0;
        __syncthreads();

        //process the bottomhorizontal border
        for(int i=0; i < repetitions; i++) {
            x = col * subBlockDim + localDepth + i*depth;
            y = (row+1) * subBlockDim - 1;
            if(!inLeftBorder()){
                changed = 1;
                __syncthreads();
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y-1, height));
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y-1, height));
            }
            if(!inRightBorder()){
                changed = 1;
                __syncthreads();
                Union(dLabelsData, x * y, (x+1)*(y+1));
            }
        }
        //process the right vertical border
        for(int i=0; i < repetitions; i++) {
            x = (col+1) * subBlockDim - 1;
            y = row * subBlockDim + localDepth + i * depth;
            if(!inTopBorder()){
                changed = 1;
                __syncthreads();
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y-1, height));
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y-1, height));
            }
            if(!inBottomBorder()){
                changed = 1;
                __syncthreads();
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y-1, height));
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y-1, height));
            }
        }
        __syncthreads();
        if(changed == 0) break; //no changes −> the tiles are merged
        __syncthreads();
    }
}

__device__ bool inLeftBorder(){

    return (threadIdx.x == 0 && blockIdx.x == 0);
}

__device__ bool inRightBorder(){

    return (blockIdx.x == (blockDim.x - 1) && threadIdx.x == BLOCK_WIDTH-1);
}

__device__ bool inTopBorder(){

    return (threadIdx.y == 0 && blockIdx.y == 0);
}

__device__ bool inBottomBorder(){

    return (blockIdx.y == (blockDim.y - 1) && threadIdx.y == BLOCK_HEIGHT-1);
}

__device__ int globalAddr(const int x, const int y, const int height){
    return x * height + y;
}

__device__ int findRoot(int equivalenceArray[], int elementAddress){

    while(equivalenceArray[elementAddress] != elementAddress)
        elementAddress = equivalenceArray[elementAddress];
    return elementAddress;
}

__device__ void Union(int equivalenceArray[], const int elementAddress0, const int elementAddress1){

    int root0 = findRoot(equivalenceArray, elementAddress0);
    int root1 = findRoot(equivalenceArray, elementAddress1);
    //connect an equivalence tree with a higher label to the tree with a lower label
    if(root0 < root1) 
        equivalenceArray[root1] = root0;
    else if(root1 < root0) 
        equivalenceArray[root0] = root1;
}
