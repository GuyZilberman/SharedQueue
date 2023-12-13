#include <iostream>
#include <cuda_runtime.h>

__global__ void printHello(int threadId) {
    for (int i = 0; i < 100000; ++i) {
        printf("Hello from thread %d\n", threadId);
    }
}

int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch kernel in stream 1
    printHello<<<1, 1, 0, stream1>>>(1);

    // Launch kernel in stream 2
    printHello<<<1, 1, 0, stream2>>>(2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
