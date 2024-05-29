#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>

// Kernel function to display 3D grid and thread information
__global__ void display_3Dgrid_threads(int* array_nums) {
    // Calculate thread ID within the block
    int tid = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Calculate the total number of threads in a block
    int no_of_threads_in_a_block = blockDim.x * blockDim.y * blockDim.z;

    // Calculate the block offset for 1D indexing
    int block_offset = no_of_threads_in_a_block * blockIdx.x;

    // Calculate the number of threads in a row of blocks
    int no_of_threads_in_a_row = no_of_threads_in_a_block * gridDim.x;

    // Calculate the row offset for 2D indexing
    int row_offset = no_of_threads_in_a_row * blockIdx.y;

    // Calculate the number of threads in an xy-plane of blocks
    int no_of_threads_in_xy_plane = no_of_threads_in_a_block * gridDim.x * gridDim.y;

    // Calculate the z offset for 3D indexing
    int z_offset = no_of_threads_in_xy_plane * blockIdx.z;

    // Calculate the global index for the current thread
    int gid = tid + block_offset + row_offset + z_offset;

    // Print the thread ID, global ID, and the value at the global ID in the array
    printf("tid : %d , gid : %d , value : %d \n", tid, gid, array_nums[gid]);
}

int main() {
    // Define the size of the array and calculate the byte size
    int size = 64; // Number of elements in the array
    int byte_size = size * sizeof(int);

    // Allocate memory for the host input array
    int* h_input;
    h_input = (int*)malloc(byte_size);

    // Seed for random number generation
    time_t t;
    srand((unsigned)time(&t));

    // Initialize the host array with random values
    for (int i = 0; i < size; i++) {
        h_input[i] = (int)(rand() & 0xff);
    }

    // Allocate memory for the device input array
    int* d_input;
    cudaMalloc((void**)&d_input, byte_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(2, 2, 2); // Block with 2x2x2 threads
    dim3 grid(4, 4, 4);  // Grid with 4x4x4 blocks

    // Launch the kernel function
    display_3Dgrid_threads<<<grid, block>>>(d_input);

    // Synchronize the device to ensure all threads have completed execution
    cudaDeviceSynchronize();

    // Free the device and host memory
    cudaFree(d_input);
    free(h_input);

    // Reset the device before exiting
    cudaDeviceReset();
    return 0;
}
