#include "common.cuh"
#include "shared_queue.cuh"
#include "/etc/pliops/store_lib_expo.h"
#include <random>

#define NO_OPTIONS 0
#define NUM_ITERATIONS QUEUE_SIZE
#define READ_START_ID NUM_ITERATIONS
#define READ_END_ID 2*NUM_ITERATIONS-1

#define GPU_TURN 0
#define CPU_TURN 1

__global__
void GPU_thread(cuda::atomic<int>* flag) {
    printf("Hello from GPU_thread\n");
    for (int i = 0; i < 5; i++)
    {
        while(flag->load() != GPU_TURN){}
        printf("GPU_thread: %d\n", i);
        flag->store(CPU_TURN);
    }
    printf("Bye from GPU_thread\n");
}

void server_func(cuda::atomic<int>* flag){  
    printf("Hello from server_func\n");
    for (int i = 0; i < 5; i++)
    {
        while(flag->load() != CPU_TURN){}
        printf("server_func: %d\n", i);
        flag->store(GPU_TURN);
    }
    printf("Bye from server_func\n");
}

int main() {
    cuda::atomic<int>* p_flag;

    // 1. Two queues - Allocate memory that is shared by the CPU and the GPU
	cudaHostAlloc((void **)&p_flag, sizeof(cuda::atomic<int>), cudaHostAllocMapped);

    new (p_flag) cuda::atomic<int>;
    *p_flag = CPU_TURN;
    
    // Launch a server thread
    std::thread server_thread(server_func, p_flag);
    server_thread.detach();
    
    std::cout << "A" << std::endl;
    // Launch the two kernels
    GPU_thread<<<1,1>>>(p_flag);
    std::cout << "B" << std::endl;

    std::cout << "Before" << std::endl;
    cudaDeviceSynchronize();
    std::cout << "After" << std::endl;


	cudaFreeHost(p_flag);
    return 0;
}