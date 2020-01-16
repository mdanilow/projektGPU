
//__device__ bool inLeftBorder();
//__device__ bool inRightBorder();
//__device__ bool inTopBorder();
//__device__ bool inBottomBorder();
__device__ bool inRange(const int x, const int y, const int x_range, const int y_range);
__device__ int globalAddr(const int x, const int y, const int x_size);
__device__ int findRoot(int equivalenceArray[], int elementAddress);
__device__ void Union(int equivalenceArray[], const int elementAddress0, const int elementAddress1, int& changed);


__global__ void mergeTiles(
        int* dLabelsData,
        const int x_size,
        const int y_size){

    __shared__ int changed;
    
    const int subBlockDim = 16;//x_size/blockDim.x;
    const int repetitions = int(subBlockDim/blockDim.z);
    const int subBlock_x = blockIdx.x*blockDim.x + threadIdx.x;
    const int subBlock_y = blockIdx.y*blockDim.y + threadIdx.y;
    int x, y = 0;

    do {
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) changed = 0;
        __syncthreads();
        
        for(int i=0; i < repetitions; i++) {
            x = subBlock_x*subBlockDim + threadIdx.z + i*blockDim.z;
            y = (subBlock_y + 1)*subBlockDim - 1;
            if(inRange(x, y, x_size, y_size)) {
                //if(!inLeftBorder()) {
                if(inRange(x-1, y+1, x_size, y_size)) {
                    Union(dLabelsData, globalAddr(x, y, x_size), globalAddr(x-1, y+1, x_size), changed);
                }
                if(inRange(x, y+1, x_size, y_size)) {
                    Union(dLabelsData, globalAddr(x, y, x_size), globalAddr(x, y+1, x_size), changed);
                }
                //if(!inRightBorder()) {
                if(inRange(x+1, y+1, x_size, y_size)) {
                    Union(dLabelsData, globalAddr(x, y, x_size), globalAddr(x+1, y+1, x_size), changed);
                }
            }
        }
        for(int i=0; i < repetitions; i++) {
            x = (subBlock_x + 1)*subBlockDim -1;
            y = subBlock_y*subBlockDim + threadIdx.z + i*blockDim.z;
            
            if(inRange(x, y, x_size, y_size)){ 
                //if(!inTopBorder()){
                if(inRange(x+1, y-1, x_size, y_size)) {
                    Union(dLabelsData, globalAddr(x, y, x_size), globalAddr(x+1, y-1, x_size), changed);
                }
                if(inRange(x+1, y, x_size, y_size)) {
                    Union(dLabelsData, globalAddr(x, y, x_size), globalAddr(x+1, y, x_size), changed);
                }
                //if(!inBottomBorder()) {
                if(inRange(x+1, y+1, x_size, y_size)) {
                    Union(dLabelsData, globalAddr(x, y, x_size), globalAddr(x+1, y+1, x_size), changed);
                }
            }
        }
        __syncthreads();
    } while(changed);
}

__device__ bool inRange(const int x, const int y, const int x_range, const int y_range) {
	return x >= 0 and y >= 0 and x < x_range and y < y_range;
}

/*
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
*/
__device__ int globalAddr(const int x, const int y, const int x_size){
    return x + y * x_size;
}

__device__ int findRoot(int equivalenceArray[], int elementAddress){

    while(equivalenceArray[elementAddress] != elementAddress)
        elementAddress = equivalenceArray[elementAddress];
    return elementAddress;
}

__device__ void Union(int equivalenceArray[], const int elementAddress0, const int elementAddress1, int& changed){

        int root0 = findRoot(equivalenceArray, elementAddress0);
        int root1 = findRoot(equivalenceArray, elementAddress1);
        if(root0 < root1) {
            equivalenceArray[root1] = root0;
            //atomicMin(equivalenceArray + root1, root0);
            changed = 1;
        }
        else if(root1 < root0){
            equivalenceArray[root0] = root1;
            //atomicMin(equivalenceArray + root0, root1);
            changed = 1;
        }
}
