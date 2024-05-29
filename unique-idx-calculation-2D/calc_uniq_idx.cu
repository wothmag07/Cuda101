#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Kernel function to print the thread index and corresponding array element
__global__ void assign_threadtoArrayEle(int *num) {
    int val = threadIdx.x;
    printf("ThreadIdx : %d, Value : %d \n", val, num[val]);
}

// Kernel function to calculate global index in 1D grid and print details
__global__ void unique_calc_gid_1d(int *num) {
    int tid = threadIdx.x;                  // Thread index within the block
    int offset = blockDim.x * blockIdx.x;   // Offset due to block index
    int gid = tid + offset;                 // Global index

    printf("ThreadIdx : %d | offset : %d | gid : %d | value : %d \n", tid, offset, gid, num[gid]);
}

// Kernel function to calculate global index in 2D grid and print details
__global__ void unique_calc_gid_2d(int* num) {
    int tid = threadIdx.x;                                // Thread index within the block
    int block_offset = blockDim.x * blockIdx.x;           // Offset due to block index in x-direction
    int row_offset = gridDim.x * blockDim.x * blockIdx.y; // Offset due to block index in y-direction
    int gid = tid + row_offset + block_offset;            // Global index

    printf("ThreadIdx : %d | row_offset : %d | block_offset : %d | gid : %d | value : %d \n", tid, row_offset, block_offset, gid, num[gid]);
}

// Kernel function to calculate global index in 2D grid with 2D blocks and print details
__global__ void unique_calc_gid_2d_2d(int* num) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;     // Thread index within the block in 2D
    int num_threads_in_a_block = blockDim.x * blockDim.y; // Total number of threads in a block
    int block_offset = num_threads_in_a_block * blockIdx.x;// Offset due to block index in x-direction

    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x; // Total number of threads in a row (all blocks in a row)
    int row_offset = num_threads_in_a_row * blockIdx.y;            // Offset due to block index in y-direction

    int gid = tid + block_offset + row_offset;                    // Global index

    printf("blockIdx.x : %d | blockIdx.y : %d | tid : %d | gid : %d | data : %d \n", blockIdx.x, blockIdx.y, tid, gid, num[gid]);
}

int main() {
    // Define and initialize host array
    int arraySize = 16;
    int array_byteSize = sizeof(int) * arraySize;
    int hostData[] = {21, 34, 12, 99, 1, 67, 82, 9, 34, 53, 2, 31, 76, 54, 44, 11};

    // Print host array
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", hostData[i]);
    }
    printf("\n \n");

    // Allocate memory on device
    int *deviceData;
    cudaMalloc((void**) &deviceData, array_byteSize);
    cudaMemcpy(deviceData, hostData, array_byteSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(4); // 1D block of 4 threads
    dim3 grid(2,2); // 2D grid of 2x2 blocks

    // Launch the kernel with different configurations
    //assign_threadtoArrayEle<<<grid, block>>>(deviceData);  // 1D grid and 1D block
    //unique_calc_gid_1d<<<grid, block>>>(deviceData);       // 1D grid and 1D block
    //unique_calc_gid_2d<<<grid, block>>>(deviceData);       // 2D grid and 1D block
    unique_calc_gid_2d_2d<<<grid, block>>>(deviceData);      // 2D grid and 2D block

    // Synchronize and reset the device
    cudaDeviceSynchronize();  
    cudaDeviceReset();

    return 0;
}
