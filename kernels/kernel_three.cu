__device__ int findRoot(const int equivalenceArray[], int elementAddress)
__device__ int globalAddr(const int x, const int y, const int height)


__global__ void flatten(int* dLabelsData, const int height, const int width, const int BLOCK_WIDTH){

    int subBlockY = blockIdx.y*blockDim.y + threadIdx.y;
    int subBlockX = blockIdx.x*blockDim.x + threadIdx.x;

    x = subBlockX * BLOCK_WIDTH + threadIdx.z;
    y = (subBlockY+1) * BLOCK_WIDTH - 1;
    int gAddr = globalAddr(x, y, height);
    dLabelsData[gAddr] = findRoot(dLabelsData, gAddr);
    gAddr = globalAddr(x, y+1, height)
    dLabelsData[gAddr] = findRoot(dLabelsData, gAddr);

    y = subBlockY * BLOCK_WIDTH + threadIdx.z;
    x = (subBlockX+1) * BLOCK_WIDTH - 1;
    gAddr = globalAddr(x, y, height);
    dLabelsData[gAddr] = findRoot(dLabelsData, gAddr);
    gAddr = globalAddr(x+1, y, height);
    dLabelsData[gAddr] = findRoot(dLabelsData, gAddr);
}


__device__ int findRoot(const int equivalenceArray[], int elementAddress){

    while(equivalenceArray[elementAddress] != elementAddress)
        elementAddress = equivalenceArray[elementAddress];
    return elementAddress;
}


__device__ int globalAddr(const int x, const int y, const int height){
    return x * height + y;
}