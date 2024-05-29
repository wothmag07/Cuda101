#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Kernel function to print thread, block, and grid information
__global__ void print_threadInfo() {
    // Calculate the thread's unique index within the grid in each dimension
    int threadId_x = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId_y = threadIdx.y + blockIdx.y * blockDim.y;
    int threadId_z = threadIdx.z + blockIdx.z * blockDim.z;

    // Get the block's index in each dimension
    int blockId_x = blockIdx.x;
    int blockId_y = blockIdx.y;
    int blockId_z = blockIdx.z;

    // Get the grid's dimensions in each dimension
    int gridId_x = gridDim.x;
    int gridId_y = gridDim.y;
    int gridId_z = gridDim.z;

    // Print the calculated thread, block, and grid information
    printf("Thread ID: (%d, %d, %d) | Block ID: (%d, %d, %d) | Grid ID: (%d, %d, %d) \n",
           threadId_x, threadId_y, threadId_z, blockId_x, blockId_y, blockId_z, gridId_x, gridId_y, gridId_z);
}

int main() {
    // Define the dimensions of the grid and blocks
    int nx = 4, ny = 4, nz = 4; // Define grid dimensions (4x4x4)
    dim3 block(2, 2, 2);        // Define block dimensions (2x2x2), total 8 threads per block

    // Calculate grid dimensions based on block dimensions
    dim3 grid(nx / block.x, ny / block.y, nz / block.z);

    // Launch the kernel function on the GPU
    print_threadInfo<<<grid, block>>>();

    // Synchronize the device to ensure all threads have completed execution
    cudaDeviceSynchronize();

    // Reset the device before exiting
    cudaDeviceReset();

    return 0;
}
