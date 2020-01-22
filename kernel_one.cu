#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

enum NEIGH_TYPE {NEIGH_FOUR = 0, NEIGH_EIGHT = 1};

__device__ int getNeighboursLocalIndexes(int neighbours[], int nType);
__device__ int getLocalIndex(int localRow, int localCol);
__device__ bool inLocalBorder();
__device__ int findRoot(int equivalenceMatrix[], int elementIndex);
__device__ bool threadInImage(int height, int width);
__device__ int localAddrToGlobal(int label, int imHeight);


__global__ void localCCL(const int* input, int* output, const int height, const int width){

    __shared__ int segments[BLOCK_WIDTH * BLOCK_HEIGHT];
    __shared__ int labels[BLOCK_WIDTH * BLOCK_HEIGHT];
    __shared__ int changed;
    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int localRow = threadIdx.x;
    int localCol = threadIdx.y;
    int localIndex = localCol * blockDim.y + localRow;
    int globalIndex = col * height + row;
    int newLabel;
    int nType = NEIGH_EIGHT;

    // load corresponding image tile to shared memory
    segments[localIndex] = input[globalIndex];

    // clear borders in every tile
    // if(inLocalBorder()){
    //     segments[localIndex] = 0;
    // }

    __syncthreads();
    int label = localIndex;
    int neighboursIndexes[8];
    int numOfNeighbours;

    if(threadInImage(height, width)){

        while(1){

            labels[localIndex] = label;

            if(localRow == 0 && localCol == 0)
                changed = 0;

            __syncthreads();
            newLabel = label;

            numOfNeighbours = getNeighboursLocalIndexes(neighboursIndexes, nType);
            
            for(int n = 0; n < numOfNeighbours; n++)
                if(segments[localIndex] == segments[neighboursIndexes[n]])
                    newLabel = min(newLabel, labels[neighboursIndexes[n]]);
            
            __syncthreads();

            if(newLabel < label){

                atomicMin(labels + label, newLabel);
                changed = 1;
            }

            __syncthreads();

            if(changed == 0)
                break;

            label = findRoot(labels, label);
            __syncthreads();
        }
    }

    output[globalIndex] = localAddrToGlobal(label, height);

    // if(input[globalIndex] == 0)
    //     output[globalIndex] = 0; 
}


__device__ int localAddrToGlobal(int label, int imHeight){

    int row = blockIdx.y*blockDim.y + label/BLOCK_WIDTH;
    int col = blockIdx.x*blockDim.x + label%16;
    
    return col * imHeight + row;
}


//returns length of neighbours list
__device__ int getNeighboursLocalIndexes(int neighbours[], int nType){

    int localRow = threadIdx.x;
    int localCol = threadIdx.y;
    int length;

    if(nType == NEIGH_FOUR){

        if(localRow == 0){

            if(localCol == 0){

                neighbours[0] = getLocalIndex(localRow, localCol + 1);
                neighbours[1] = getLocalIndex(localRow + 1, localCol);
                length = 2;
            }

            else if(localCol == BLOCK_HEIGHT - 1){

                neighbours[0] = getLocalIndex(localRow, localCol - 1);
                neighbours[1] = getLocalIndex(localRow + 1, localCol);
                length = 2; 
            }

            else{

                neighbours[0] = getLocalIndex(localRow, localCol - 1);
                neighbours[1] = getLocalIndex(localRow, localCol + 1);
                neighbours[3] = getLocalIndex(localRow + 1, localCol);
                length = 3;
            }
        }

        else if(localRow == BLOCK_HEIGHT-1){

            if(localCol == 0){

                neighbours[0] = getLocalIndex(localRow, localCol + 1);
                neighbours[1] = getLocalIndex(localRow - 1, localCol);
                length = 2;
            }

            else if(localCol == BLOCK_HEIGHT - 1){

                neighbours[0] = getLocalIndex(localRow, localCol - 1);
                neighbours[1] = getLocalIndex(localRow - 1, localCol);
                length = 2; 
            }

            else{

                neighbours[0] = getLocalIndex(localRow, localCol - 1);
                neighbours[1] = getLocalIndex(localRow, localCol + 1);
                neighbours[2] = getLocalIndex(localRow - 1, localCol);
                length = 3;
            }
        }

        else if(localCol == 0){

            neighbours[0] = getLocalIndex(localRow - 1, localCol);
            neighbours[1] = getLocalIndex(localRow, localCol + 1);
            neighbours[2] = getLocalIndex(localRow + 1, localCol);
            length = 3;
        }

        else if(localCol == BLOCK_WIDTH-1){

            neighbours[0] = getLocalIndex(localRow - 1, localCol);
            neighbours[1] = getLocalIndex(localRow, localCol - 1);
            neighbours[2] = getLocalIndex(localRow + 1, localCol);
            length = 3;
        }

        else{

            neighbours[0] = getLocalIndex(localRow - 1, localCol);
            neighbours[1] = getLocalIndex(localRow, localCol + 1);
            neighbours[2] = getLocalIndex(localRow + 1, localCol);
            neighbours[3] = getLocalIndex(localRow, localCol - 1);
            length = 4;
        }
    }


    else if(nType == NEIGH_EIGHT){

        if(localRow == 0){

            if(localCol == 0){
                neighbours[0] = getLocalIndex(localRow, localCol + 1);
                neighbours[1] = getLocalIndex(localRow + 1, localCol);
                neighbours[2] = getLocalIndex(localRow + 1, localCol + 1);
                length = 3;
            }

            else if(localCol == BLOCK_WIDTH-1){
                neighbours[0] = getLocalIndex(localRow, localCol - 1);
                neighbours[1] = getLocalIndex(localRow + 1, localCol);
                neighbours[2] = getLocalIndex(localRow + 1, localCol - 1);
                length = 3;
            }

            else{
                neighbours[0] = getLocalIndex(localRow + 1, localCol - 1);
                neighbours[1] = getLocalIndex(localRow + 1, localCol);
                neighbours[2] = getLocalIndex(localRow + 1, localCol + 1);
                neighbours[3] = getLocalIndex(localRow, localCol - 1);
                neighbours[4] = getLocalIndex(localRow, localCol + 1);
                length = 5;
            }
        }

        else if(localRow == BLOCK_HEIGHT-1){
            
            if(localCol == 0){
                neighbours[0] = getLocalIndex(localRow, localCol + 1);
                neighbours[1] = getLocalIndex(localRow - 1, localCol);
                neighbours[2] = getLocalIndex(localRow - 1, localCol + 1);
                length = 3;
            }

            else if(localCol == BLOCK_WIDTH-1){
                neighbours[0] = getLocalIndex(localRow, localCol - 1);
                neighbours[1] = getLocalIndex(localRow - 1, localCol);
                neighbours[2] = getLocalIndex(localRow - 1, localCol - 1);
                length = 3;
            }

            else{
                neighbours[0] = getLocalIndex(localRow - 1, localCol - 1);
                neighbours[1] = getLocalIndex(localRow - 1, localCol);
                neighbours[2] = getLocalIndex(localRow - 1, localCol + 1); 
                neighbours[3] = getLocalIndex(localRow, localCol - 1);
                neighbours[4] = getLocalIndex(localRow, localCol + 1);
                length = 5;
            }
        }

        else if(localCol == 0){

            neighbours[0] = getLocalIndex(localRow - 1, localCol);
            neighbours[1] = getLocalIndex(localRow - 1, localCol + 1);
            neighbours[2] = getLocalIndex(localRow, localCol + 1); 
            neighbours[3] = getLocalIndex(localRow + 1, localCol);
            neighbours[4] = getLocalIndex(localRow + 1, localCol + 1);
            length = 5;
        }

        else if(localCol == BLOCK_WIDTH-1){

            neighbours[0] = getLocalIndex(localRow - 1, localCol);
            neighbours[1] = getLocalIndex(localRow - 1, localCol - 1);
            neighbours[2] = getLocalIndex(localRow, localCol - 1); 
            neighbours[3] = getLocalIndex(localRow + 1, localCol);
            neighbours[4] = getLocalIndex(localRow + 1, localCol - 1);
            length = 5;
        }

        else{

            neighbours[0] = getLocalIndex(localRow - 1, localCol - 1);
            neighbours[1] = getLocalIndex(localRow - 1, localCol);
            neighbours[2] = getLocalIndex(localRow - 1, localCol + 1);
            neighbours[3] = getLocalIndex(localRow, localCol + 1);
            neighbours[4] = getLocalIndex(localRow + 1, localCol + 1);
            neighbours[5] = getLocalIndex(localRow + 1, localCol);
            neighbours[6] = getLocalIndex(localRow + 1, localCol - 1);
            neighbours[7] = getLocalIndex(localRow, localCol - 1);

            length = 8;
        }
    }

    return length;
}


__device__ int getLocalIndex(int localRow, int localCol){

    return localCol * blockDim.y + localRow;
}


__device__ bool inLocalBorder(){

    return (threadIdx.x == 0 || threadIdx.x == BLOCK_WIDTH-1 || threadIdx.y == 0 || threadIdx.y == BLOCK_HEIGHT-1);
}


__device__ bool threadInImage(int height, int width){

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    return (row >= 0 || row <= height-1 || col >= 0 || col <= width-1);
} 


__device__ int findRoot(int equivalenceMatrix[], int elementIndex){

    while(equivalenceMatrix[elementIndex] != elementIndex)
        elementIndex = equivalenceMatrix[elementIndex];

    return elementIndex;
}
