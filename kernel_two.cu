#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

__device__ bool inLeftBorder();
__device__ bool inRightBorder();
__device__ bool inTopBorder();
__device__ bool inBottomBorder();
__device__ int globalAddr(const int x, const int y, const int height);
__device__ int findRoot(int equivalenceArray[], int elementAddress);
__device__ void Union(int* equivalenceArray, const int elementAddress0, const int elementAddress1, int& changed);


__global__ void mergeTiles(
        const double* dSegData,
        int* dLabelsData,
        const int height,
        const int width){

    __shared__ int changed;

    int x, y = 0;
    
    int repetitions = int(blockIdx.z/blockDim.z);

    while(1) {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) changed = 0;
        __syncthreads();

        for(int i=0; i < repetitions; i++) {
            x = blockIdx.x*blockDim.x + threadIdx.z + i*blockDim.z;
            y = (blockIdx.y+1)*blockDim.y - 1;
            if(!inLeftBorder()) {
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x-1, y+1, height), changed);
            }
            Union(dLabelsData, globalAddr(x, y, height), globalAddr(x, y+1, height), changed);
            if(!inRightBorder()) {
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x+1, y+1, height), changed);
            }
        }
        for(int i=0; i < repetitions; i++) {
            x = (blockIdx.x+1)*blockDim.x -1;
            y = blockIdx.y * blockDim.y + threadIdx.z + i*blockDim.z;
            if(!inTopBorder()){
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x+1, y-1, height), changed);
            }
            Union(dLabelsData, globalAddr(x, y, height), globalAddr(x+1, y, height), changed);
            if(!inBottomBorder()) {
                Union(dLabelsData, globalAddr(x, y, height), globalAddr(x+1, y+1, height), changed);
            }
        }
        __syncthreads();
        if(changed == 0) break;
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

__device__ void Union(int* equivalenceArray, const int elementAddress0, const int elementAddress1, int &changed){

    int root0 = findRoot(equivalenceArray, elementAddress0);
    int root1 = findRoot(equivalenceArray, elementAddress1);
    if(root0 < root1) {
        //equivalenceArray[root1] = root0;
        atomicMin(equivalenceArray + root1, root0);
        changed = 1;
    }
    else if(root1 < root0){
        //equivalenceArray[root0] = root1;
        atomicMin(equivalenceArray + root0, root1);
        changed = 1;
    }
}
