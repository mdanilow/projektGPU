
__device__ bool inLeftBorder();
__device__ bool inRightBorder(int imWidth, int BLOCK_WIDTH, int batchNum);
__device__ bool inTopBorder();
__device__ bool inBottomBorder(int imHeight, int BLOCK_WIDTH, int batchNum);
__device__ int globalAddr(const int x, const int y, const int height);
__device__ int findRoot(int equivalenceArray[], int elementAddress);
__device__ void Union(int equivalenceArray[], const int segmentsArray[], const int elementAddress0, const int elementAddress1, int* changedPtr);


__global__ void mergeTiles(
        const int* dSegData,
        int* dLabelsData,
        const int height,
        const int width,
        const int BLOCK_WIDTH){

    __shared__ int changed; //shared memory used to check whether the solution is final or not

    int subBlockY = blockIdx.y*blockDim.y + threadIdx.y;
    int subBlockX = blockIdx.x*blockDim.x + threadIdx.x;

    int x, y = 0;
    int repetitions = BLOCK_WIDTH / blockDim.z; //how many times are the thread reused for the given subblock?
    
    // printf("blockIdx.x: %d\nblockIdx.y: %d\nthreadIdx.x: %d\nthreadIdx.y: %d\nsubBlockX: %d\nsubBlockY: %d", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, subBlockX, subBlockY);
    
    while(1) {

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) 
            changed = 0;
        __syncthreads();

        //process the bottomhorizontal border
        for(int i = 0; i < repetitions; i++){
            //pixel coordinates
            x = subBlockX * BLOCK_WIDTH + threadIdx.z + i*blockDim.z;
            y = (subBlockY+1) * BLOCK_WIDTH - 1;

            if(globalAddr(x+1, y+1, height) < 0 || globalAddr(x+1, y+1, height) > width*height-1)
                printf("width: %d, height: %d, horizontal %d\n", width, height, globalAddr(x, y, height));  

            if(!inLeftBorder())
                Union(dLabelsData, dSegData, globalAddr(x, y, height), globalAddr(x-1, y+1, height), &changed);

            Union(dLabelsData, dSegData, globalAddr(x, y, height), globalAddr(x, y+1, height), &changed);  

            if(!inRightBorder(width, BLOCK_WIDTH, i))
                Union(dLabelsData, dSegData, globalAddr(x, y, height), globalAddr(x+1, y+1, height), &changed);
        }

        //process the right vertical border
        for(int i = 0; i < repetitions; i++){
            //pixel coordinates
            y = subBlockY * BLOCK_WIDTH + threadIdx.z + i*blockDim.z;
            x = (subBlockX+1) * BLOCK_WIDTH - 1;

            if(globalAddr(x, y, height) < 0 || globalAddr(x, y, height) > width*height-1)
                printf("vertical %d\n", globalAddr(x, y, height));  

            if(!inTopBorder())
                Union(dLabelsData, dSegData, globalAddr(x, y, height), globalAddr(x+1, y-1, height), &changed);

            Union(dLabelsData, dSegData, globalAddr(x, y, height), globalAddr(x+1, y, height), &changed);

            if(!inBottomBorder(height, BLOCK_WIDTH, i))
                Union(dLabelsData, dSegData, globalAddr(x, y, height), globalAddr(x+1, y+1, height), &changed);
        }

            __syncthreads();
            if(changed == 0) break; //no changes −> the tiles are merged
            __syncthreads();    
    }
}

__device__ bool inLeftBorder(){

    return (threadIdx.x == 0 && blockIdx.x == 0);
}

__device__ bool inRightBorder(int imWidth, const int BLOCK_WIDTH, int batchNum){

    int subBlockX = blockIdx.x*blockDim.x + threadIdx.x;
    int x = subBlockX * BLOCK_WIDTH + threadIdx.z + batchNum*blockDim.z;

    return (x >= imWidth);
}

__device__ bool inTopBorder(){

    return (threadIdx.y == 0 && blockIdx.y == 0);
}

__device__ bool inBottomBorder(int imHeight, const int BLOCK_WIDTH, int batchNum){

    int subBlockY = blockIdx.y*blockDim.y + threadIdx.y;
    int y = subBlockY * BLOCK_WIDTH + threadIdx.z + batchNum*blockDim.z;

    return (y == imHeight);
}

__device__ int globalAddr(const int x, const int y, const int height){
    return x * height + y;
}

__device__ int findRoot(int equivalenceArray[], int elementAddress){

    while(equivalenceArray[elementAddress] != elementAddress)
        elementAddress = equivalenceArray[elementAddress];
    return elementAddress;
}

__device__ void Union(int equivalenceArray[], const int segmentsArray[], const int elementAddress0, const int elementAddress1, int* changedPtr){

    if(segmentsArray[elementAddress0] == segmentsArray[elementAddress1]){

        int root0 = findRoot(equivalenceArray, elementAddress0);
        int root1 = findRoot(equivalenceArray, elementAddress1);
        //connect an equivalence tree with a higher label to the tree with a lower label
        if(root0 < root1){
            atomicMin(equivalenceArray + root1, root0);
            *changedPtr = 1;
        }
        else if(root1 < root0) {
            atomicMin(equivalenceArray + root0, root1);
            *changedPtr = 1;
        }
    }
}
