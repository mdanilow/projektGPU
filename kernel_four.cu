
__device__ int findRoot(const int equivalenceMatrix[], int elementIndex){

    while(equivalenceMatrix[elementIndex] != elementIndex)
        elementIndex = equivalenceMatrix[elementIndex];

    return elementIndex;
}



__global__ void finalUpdate(const int* input, int* output, const int height, const int width){

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int globalIndex = col * height + row;

    output[globalIndex] = findRoot(input, input[globalIndex]);
}




