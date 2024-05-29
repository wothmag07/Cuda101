#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>

// Kernel function for memory transfer and printing array elements without bounds checking
__global__ void mem_transfer_cpy(int* arr, int size) {
    // Calculate global index
    int gid = blockDim.x * blockIdx.x + threadIdx.x; // 1-D thread block array
    printf("tid : %d | gid : %d | value : %d \n", threadIdx.x, gid, arr[gid]);
}

// Kernel function for memory transfer and printing array elements with bounds checking
__global__ void mem_transfer_cpy_alter(int* arr, int size) {
    // Calculate global index
    int gid = blockDim.x * blockIdx.x + threadIdx.x; // 1-D thread block array

    // Check if the global index is within the bounds of the array
    if (gid < size) {
        printf("tid : %d | gid : %d | value : %d \n", threadIdx.x, gid, arr[gid]);
    }
}

int main() {
    // Define the size of the array and calculate the byte size
    int size = 140;
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
    dim3 block(32); // Block with 32 threads
    dim3 grid(5);   // Grid with 5 blocks

    // Launch the kernel function without bounds checking (commented out)
    // mem_transfer_cpy<<<grid, block>>>(d_input, size);

    // Launch the kernel function with bounds checking
    mem_transfer_cpy_alter<<<grid, block>>>(d_input, size);

    // Synchronize the device to ensure all threads have completed execution
    cudaDeviceSynchronize();

    // Free the device and host memory
    cudaFree(d_input);
    free(h_input);

    // Reset the device before exiting
    cudaDeviceReset();
    return 0;
}
