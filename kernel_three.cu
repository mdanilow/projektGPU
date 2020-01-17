__device__ int findRoot(int equivalenceArray[], int elementAddress)


__global__ void flatten(int* dLabelsData, const int height, const int width){

    int subBlockY = blockIdx.y*blockDim.y + threadIdx.y;
    int subBlockX = blockIdx.x*blockDim.x + threadIdx.x;
}


__device__ int findRoot(int equivalenceArray[], int elementAddress){

    while(equivalenceArray[elementAddress] != elementAddress)
        elementAddress = equivalenceArray[elementAddress];
    return elementAddress;
}