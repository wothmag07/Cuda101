#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Kernel function to print a greeting message
__global__ void hellocuda() {
    printf("Hello AMIGOS :)) \n");
}

int main() {
	
    // Define grid and block dimensions for a total of 64 threads (16 x 4)
    int nx = 16, ny = 4;

    // Define block dimensions: 8 threads in x-dimension and 2 threads in y-dimension
    dim3 block(8, 2, 1);

    // Calculate grid dimensions based on the block dimensions
    dim3 grid(nx / block.x, ny / block.y, 1);

    // Launch the kernel function on the GPU
    hellocuda<<<grid, block>>>();

    // Synchronize the device to ensure all threads have completed execution
    cudaDeviceSynchronize();

    // Reset the device before exiting
    cudaDeviceReset();

    return 0;
}
