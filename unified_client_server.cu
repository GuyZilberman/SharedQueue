#include "common.cuh"
#include "shared_queue.cuh"
#include "/etc/pliops/store_lib_expo.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


#include "gdrapi.h"
#include "gdrcopy_common.hpp"
#include "gdr_gpu_memalloc.cuh"

#define NO_OPTIONS 0
#define NUM_ITERATIONS 644
#define READ_START_ID NUM_ITERATIONS
#define READ_END_ID 2*NUM_ITERATIONS-1

__device__
void InitData(int* arr, size_t size, int idx) {
    // Set the first value in the vector to idx
    arr[0] = idx;
}

__global__
//void client_thread_func(LockFreeQueue<RequestMessage> *submission_queue, LockFreeQueue<ResponseMessage> *completion_queue, const int num_iterations, cudaEvent_t &start1, cudaEvent_t &end1, cudaEvent_t &start2, cudaEvent_t &end2) {
void client_thread_func(LockFreeQueue<RequestMessage> *submission_queue, LockFreeQueue<ResponseMessage> *completion_queue, const int num_iterations, float clockRatekHz) {
    uint idx = 0, request_id = 0;
    AnswerType answer = AnswerType::NONE;
    int wrong_answers = 0; //TODO guy - WRITE-READ test

    clock_t start1 = clock();

    // Send write requests
    while (idx < num_iterations){ 
        // Perform IO request
        printf("submission_queue: before push idx is %d\n", idx);
        RequestMessage req_msg;
        req_msg.cmd = CommandType::WRITE;
        req_msg.request_id = request_id++;
        InitData(req_msg.data, 256, idx); //TODO change 256 to req_mst.data's size
        req_msg.key = idx++;
        while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully

        // Immediately wait for a response
        ResponseMessage res_msg;
        while (!completion_queue->pop(res_msg)); // Busy-wait for a command to be available
        // TODO guy check about this: Optional: backoff strategy to reduce CPU usage
        answer = res_msg.answer;
        // printf("Client: Received from completion queue: %d\n", (int)answer); //TODO guy UNCOMMENT
    }
    clock_t end1 = clock();
    clock_t elapsedCycles1 = end1 - start1;
    float timeMilliseconds1 = (float)elapsedCycles1 / clockRatekHz;
    printf("Elapsed time 1: %.2f milliseconds\n", timeMilliseconds1);

    //cudaEventRecord(start2);
    clock_t start2 = clock();

    // Send read requests
    idx = 0;
    while (idx < num_iterations){
        RequestMessage req_msg;
        req_msg.cmd = CommandType::READ;
        req_msg.request_id = request_id++;
        req_msg.key = idx++;

        while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully

        // Immediately wait for a response
        ResponseMessage res_msg;
        while (!completion_queue->pop(res_msg)); // Busy-wait for a command to be available
        // TODO guy check about this: Optional: backoff strategy to reduce CPU usage
        answer = res_msg.answer;
        // printf("Client: Received from completion queue: %d\n", (int)answer); //TODO guy UNCOMMENT

        if (res_msg.data[0] != res_msg.request_id - NUM_ITERATIONS){ //TODO guy - WRITE-READ test
            printf("%d: res_msg.request_id\n", res_msg.request_id); //TODO guy - WRITE-READ test
            wrong_answers++;
        }
    }
    //cudaEventRecord(end2);
    clock_t end2 = clock();
    clock_t elapsedCycles2 = end2 - start2;
    float timeMilliseconds2 = (float)elapsedCycles2 / clockRatekHz;
    printf("Elapsed time 2: %.2f milliseconds\n", timeMilliseconds2);


    printf("---------------------\n");
    printf("wrong answers: %d\n", wrong_answers); //TODO guy - WRITE-READ test
    printf("---------------------\n");

    while (answer != AnswerType::EXIT)
    {
        // Send exit request
        RequestMessage req_msg_exit;
        req_msg_exit.cmd = CommandType::EXIT;
        req_msg_exit.request_id = ++request_id;
        while (!submission_queue->push(req_msg_exit)); // Busy-wait until the value is pushed successfully

        // Immediately wait for a response
        ResponseMessage res_msg;
        while (!completion_queue->pop(res_msg)); // Busy-wait for a command to be available
        answer = res_msg.answer;
        // printf("Client: Received from completion queue: %d\n", (int)answer); //TODO guy UNCOMMENT
        // printf("Client: data[0] from completion queue: %d\n", res_msg.data[0]); //TODO guy UNCOMMENT
    }
}

void server_func(LockFreeQueue<RequestMessage> *submission_queue, LockFreeQueue<ResponseMessage> *completion_queue, PLIOPS_DB_t plio_handle ){
    uint actual_object_size = 0;
    int ret = 0;
    RequestMessage req_msg; // TODO guy move this into the while loop
    CommandType command = CommandType::NONE;
    int idx = 0;
    
    while (command != CommandType::EXIT) {
        ResponseMessage res_msg;
        while (!submission_queue->pop(req_msg)); // Busy-wait for a value to be available
        command = req_msg.cmd;
        res_msg.request_id = req_msg.request_id;

            if (req_msg.cmd == CommandType::EXIT){
                res_msg.answer = AnswerType::EXIT;
            }
            else if (req_msg.cmd == CommandType::WRITE)
            {
                std::cout << "Received: " << req_msg.data[0] << std::endl;
                std::cout << req_msg.request_id << ": Calling PLIOPS_Put! Value: "  << req_msg.data[0] << std::endl;
                ret = PLIOPS_Put(plio_handle, &req_msg.key, sizeof(req_msg.key), &req_msg.data, sizeof(req_msg.data), NO_OPTIONS); //TODO guy look into options
                if (ret != 0) {
                    printf("PLIOPS_Put Failed ret=%d\n", ret);
                    res_msg.answer = AnswerType::FAIL;
                    res_msg.error = ret;
                }
                else
                    res_msg.answer = AnswerType::SUCCESS; // TODO guy - res_msg.answer = SUCCESS;
                //std::cout << "Finished PLIOPS_Put!" << std::endl; 
            }
            else if (req_msg.cmd == CommandType::READ)
            {
                //std::cout << "Calling PLIOPS_Get!" << std::endl;
                ret = PLIOPS_Get(plio_handle, &req_msg.key, sizeof(req_msg.key), &res_msg.data, sizeof(res_msg.data), &actual_object_size);
                if (ret != 0) {
                    printf("PLIOPS_Get Failed ret=%d\n", ret);
                    res_msg.answer = AnswerType::FAIL;
                    res_msg.error = ret;
                }
                else
                    res_msg.answer = AnswerType::SUCCESS; // TODO guy - res_msg.answer = SUCCESS;
                //std::cout << "Finished PLIOPS_Get!" << std::endl; 
                //std::cout << req_msg.request_id << ": Called PLIOPS_Get! Value: " << res_msg.data[0] << std::endl;
            }
            else
            {
                //std::cout << "Cannot perform command " << (int)req_msg.cmd << std::endl;
                res_msg.answer = AnswerType::FAIL;
                //TODO add: res_msg.error = ???;
            }
        //std::cout << idx << ": Before sending response message" << std::endl; //TODO guy UNCOMMENT
        while (!completion_queue->push(res_msg)); // Busy-wait until the value is pushed successfully
        //std::cout << idx++ << ": After sending response message" << std::endl; //TODO guy UNCOMMENT
        //std::cout << "Server sent confirmation message with the answer: " << (int)res_msg.answer << std::endl;

    }
    
}

bool storelib_init(PLIOPS_IDENTIFY_t& identify, PLIOPS_DB_t& plio_handle){
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options; //TODO guy check what each flag in the option does
    db_open_options.createIfMissing = 1;
    db_open_options.tailSizeInBytes = 0;
    // TODO guy ask ido: db_open_options.errorIfExists = ???

    int ret = PLIOPS_OpenDB(identify, &db_open_options, 0, &plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_OpenDB!" << std::endl;   
    return true;
}

bool storelib_deinit(PLIOPS_IDENTIFY_t& identify, PLIOPS_DB_t& plio_handle){
    int ret = PLIOPS_CloseDB(plio_handle);
    if (ret != 0) {
        printf("PLIOPS_CloseDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_CloseDB!" << std::endl;       

    ret = PLIOPS_DeleteDB(identify, 0);
    if (ret != 0) {
        printf("PLIOPS_DeleteDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_DeleteDB!" <<std::endl;  
    return true;
}

bool process_requests(PLIOPS_DB_t& plio_handle){
    cudaDeviceProp prop;
    CUDA_ERRCHECK(cudaGetDeviceProperties(&prop, 0));
    float clockRatekHz = prop.clockRate / 1.0f; // prop.clockRate is given in kHz
    printf("GPU Clock Rate: %.2f kHz\n", clockRatekHz);

    LockFreeQueue<RequestMessage>* h_sq_p;
    LockFreeQueue<ResponseMessage>* h_cq_p;
    CUdeviceptr d_cq_p; // NEW
    GPUMemoryManager *gpu_mm = new GPUMemoryManager(); // NEW

    // Two queues - Allocate memory that is shared by the CPU and the GPU
    CUDA_ERRCHECK(cudaHostAlloc((void **)&h_sq_p, sizeof(LockFreeQueue<RequestMessage>), cudaHostAllocMapped));
    cudaGPUMemAlloc<LockFreeQueue<ResponseMessage>>(gpu_mm, &h_cq_p, d_cq_p);


    new (h_sq_p) LockFreeQueue<RequestMessage>();
    new (h_cq_p) LockFreeQueue<ResponseMessage>();

    // Launch a server thread
    std::thread server_thread(server_func, h_sq_p, h_cq_p, plio_handle);
    server_thread.detach();
    
    // Launch the kernel
    client_thread_func<<<1,1>>>(h_sq_p, (LockFreeQueue<ResponseMessage> *)d_cq_p, NUM_ITERATIONS, clockRatekHz);
    //client_thread_func<<<1, 1>>>(h_sq_p, (LockFreeQueue<ResponseMessage> *)d_cq_p, NUM_ITERATIONS, start1, end1, start2, end2);

    CUDA_ERRCHECK(cudaDeviceSynchronize());
	CUDA_ERRCHECK(cudaFreeHost(h_sq_p));
    //CUDA_ERRCHECK(cudaFreeHost(h_cq_p));
    cudaGPUMemFree<LockFreeQueue<ResponseMessage>>(gpu_mm); //NEW
    delete(gpu_mm); //NEW
    return true;
}

int main() {
    PLIOPS_IDENTIFY_t identify = 0; //TODO guy check if I need a better identifier
    PLIOPS_DB_t plio_handle;
    if (!storelib_init(identify, plio_handle)){
        std::cout << "Storelib initialization failed. Exiting." << std::endl;       
        return 1;
    }

    if (!process_requests(plio_handle)) {
        std::cout << "Request processing failed. Exiting." << std::endl;       
        return 1;
    }

    if (!storelib_deinit(identify, plio_handle)){
        std::cout << "Storelib deinitialization failed. Exiting." << std::endl;       
        return 1;
    }
    return 0;
}