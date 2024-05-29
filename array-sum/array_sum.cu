#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>

// Kernel function to sum elements of two arrays on the GPU
__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

// Function to sum elements of two arrays on the CPU
void sum_array_cpu(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to compare elements of two arrays
void compare_elements(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            printf("Arrays have different elements \n");
            return;
        }
    }
    printf("Arrays have same elements \n");
}

int main() {
    int size = 1 << 25; // Number of elements in the arrays
    int block_size = 128; // Number of threads per block
    cudaError error;

    // Number of bytes needed to hold the element count
    int NO_OF_BYTES = size * sizeof(int);

    // Host pointers
    int* h_a, * h_b, * gpu_c, * cpu_c;

    // Allocate memory for host pointers
    h_a = (int*)malloc(NO_OF_BYTES);
    h_b = (int*)malloc(NO_OF_BYTES);
    cpu_c = (int*)malloc(NO_OF_BYTES);
    gpu_c = (int*)malloc(NO_OF_BYTES);

    // Initialize Host pointers with random values
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        h_a[i] = (int)(rand() & 0xff);
    }
    for (int i = 0; i < size; i++) {
        h_b[i] = (int)(rand() & 0xff);
    }

    // Summation on CPU
    clock_t start_clock_cpu, end_clock_cpu;
    start_clock_cpu = clock();
    sum_array_cpu(h_a, h_b, cpu_c, size);
    end_clock_cpu = clock();

    // Device pointers
    int* d_a, * d_b, * d_c;

    // Allocate memory on the device
    error = cudaMalloc((void**)&d_a, NO_OF_BYTES);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error : %s \n", cudaGetErrorString(error));
    }
    error = cudaMalloc((void**)&d_b, NO_OF_BYTES);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error : %s \n", cudaGetErrorString(error));
    }
    error = cudaMalloc((void**)&d_c, NO_OF_BYTES);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error : %s \n", cudaGetErrorString(error));
    }

    // Memory transfer from host to device
    clock_t host2device_start, host2device_end;
    host2device_start = clock();
    cudaMemcpy(d_a, h_a, NO_OF_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_OF_BYTES, cudaMemcpyHostToDevice);
    host2device_end = clock();

    // Kernel invocation
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);

    // Summation on GPU
    clock_t start_clock_gpu, end_clock_gpu;
    start_clock_gpu = clock();
    sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    end_clock_gpu = clock();

    // Memory transfer from device to host
    clock_t device2host_start, device2host_end;
    device2host_start = clock();
    cudaMemcpy(gpu_c, d_c, NO_OF_BYTES, cudaMemcpyDeviceToHost);
    device2host_end = clock();

    // Compare the results of CPU and GPU summations
    compare_elements(gpu_c, cpu_c, size);

    // Execution times
    printf("Execution CPU Time for array sum : %4.6f \n", (double)(end_clock_cpu - start_clock_cpu) / CLOCKS_PER_SEC);
    printf("Execution GPU Time for array sum : %4.6f \n", (double)(end_clock_gpu - start_clock_gpu) / CLOCKS_PER_SEC);
    printf("Host to Device Memory transfer time : %4.6f \n", (double)(host2device_end - host2device_start) / CLOCKS_PER_SEC);
    printf("Device to Host Memory transfer time : %4.6f \n", (double)(device2host_end - device2host_start) / CLOCKS_PER_SEC);
    printf("Total Execution GPU Time for array sum : %4.6f \n", (double)(device2host_end - host2device_start) / CLOCKS_PER_SEC);

    // Free memory
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    free(gpu_c);
    free(h_a);
    free(h_b);

    // Reset the device
    cudaDeviceReset();
    return 0;
}
