#include "common.cuh"
#include "shared_queue.cuh"
#include "/etc/pliops/store_lib_expo.h"
#include <random>

#define NO_OPTIONS 0
#define NUM_ITERATIONS QUEUE_SIZE
#define READ_START_ID NUM_ITERATIONS
#define READ_END_ID 2*NUM_ITERATIONS-1

__device__
void generateRandomInts(int* arr, size_t size, int idx) {
    // Set the first value in the vector to idx
    arr[0] = idx;
}

__global__
void completion_queue_thread_func(LockFreeQueue<ResponseMessage> *completion_queue) {
    printf("!!!!!!Hello from completion_queue_thread_func!!!!!!!\n");

    AnswerType answer = AnswerType::NONE;
    int wrong_answers = 0; //TODO guy - WRITE-READ test

    printf("completion_queue_thread_func: Before while loop\n"); //TODO guy DELETE
    int idx = 0;
    while (answer != AnswerType::EXIT) {
        printf("completion_queue_thread_func: idx is %d\n", idx); //TODO guy DELETE
        ResponseMessage res_msg;
        while (!completion_queue->pop(res_msg)); // Busy-wait for a command to be available
        // TODO guy check about this: Optional: backoff strategy to reduce CPU usage
        answer = res_msg.answer;
        printf("Client: Received from completion queue: %d\n", (int)answer);

        if (res_msg.request_id >= READ_START_ID && res_msg.request_id <= READ_END_ID){ //TODO guy - WRITE-READ test
            if (res_msg.data[0] != res_msg.request_id - NUM_ITERATIONS){
                printf("%d: Mistake found!", res_msg.request_id);
                wrong_answers++;
            }
        }
        idx++;
    }
    printf("wrong_answers: %d\n", wrong_answers);

}

__global__
void submission_queue_thread_func(LockFreeQueue<RequestMessage> *submission_queue, const int num_iterations, const int value_size) {
    printf("Hello from submission_queue_thread_func\n");

    uint idx = 0, request_id = 0;

    // Send 10000 Write requests
    while (idx < num_iterations){
        printf("submission_queue_thread_func: before push idx is %d\n", idx);
        RequestMessage req_msg;
        req_msg.cmd = CommandType::WRITE;
        req_msg.request_id = request_id++;
        generateRandomInts(req_msg.data, 256, idx);
        req_msg.key = idx;

        while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully
        printf("submission_queue_thread_func: after push idx is %d\n", idx);
        idx++;
    }

    // Send 10000 Read requests
    // idx = 0;
    // while (idx < num_iterations){
    //     printf("submission_queue_thread_func: idx is %d\n", idx);
    //     RequestMessage req_msg;
    //     req_msg.cmd = CommandType::READ;
    //     req_msg.request_id = request_id++;
    //     req_msg.key = idx++;

    //     while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully
    // }

    // Send exit request
    RequestMessage req_msg_exit;
    req_msg_exit.cmd = CommandType::EXIT;
    req_msg_exit.request_id = ++request_id;
    printf("NOT YET Bye from submission_queue_thread_func\n");
    while (!submission_queue->push(req_msg_exit)); // Busy-wait until the value is pushed successfully
    printf("Bye from submission_queue_thread_func\n");
}

void server_func(LockFreeQueue<RequestMessage> *submission_queue, LockFreeQueue<ResponseMessage> *completion_queue, sem_t* p_server_semaphore, PLIOPS_DB_t plio_handle ){
    uint actual_object_size = 0;
    int ret = 0;
    RequestMessage req_msg; // TODO guy move this into the while loop
    CommandType command = CommandType::NONE;
    int idx = 0;
    // Signal that initialization is done
    sem_post(p_server_semaphore);
    
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
        std::cout << idx << ": Before sending response message" << std::endl;
        while (!completion_queue->push(res_msg)); // Busy-wait until the value is pushed successfully
        std::cout << idx++ << ": After sending response message" << std::endl;
        //std::cout << "Server sent confirmation message with the answer: " << (int)res_msg.answer << std::endl;

    }
    
}

int main() {
    // Start of storelib init
    PLIOPS_IDENTIFY_t identify = 0; //TODO guy check if I need a better identifier
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options; //TODO guy check what each flag in the option does
    PLIOPS_DB_t plio_handle;
    db_open_options.createIfMissing = 1;
    db_open_options.tailSizeInBytes = 0;
    // TODO guy ask ido: db_open_options.errorIfExists = ???

    std::cout << "Calling PLIOPS_OpenDB!" << std::endl;       
    int ret = PLIOPS_OpenDB(identify, &db_open_options, 0, &plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_OpenDB!" << std::endl;   


    sem_t server_semaphore;
    sem_init(&server_semaphore, 0, 0); // 0 - shared between threads of a process, 0 - initial value

    // Create two CUDA streams - one for every queue
    cudaStream_t submissionQueueStream, completionQueueStream;
    cudaStreamCreate(&submissionQueueStream);
    cudaStreamCreate(&completionQueueStream);

    LockFreeQueue<RequestMessage>* h_sq_p;
    LockFreeQueue<ResponseMessage>* h_cq_p;

    // 1. Two queues - Allocate memory that is shared by the CPU and the GPU
	cudaHostAlloc((void **)&h_sq_p, sizeof(LockFreeQueue<RequestMessage>), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_cq_p, sizeof(LockFreeQueue<ResponseMessage>), cudaHostAllocMapped);

    new (h_sq_p) LockFreeQueue<RequestMessage>();
    new (h_cq_p) LockFreeQueue<ResponseMessage>();

    // Launch a server thread
    std::thread server_thread(server_func, h_sq_p, h_cq_p, &server_semaphore, plio_handle);
    server_thread.detach();

    // Wait for the server to signal its initialization is done
    sem_wait(&server_semaphore);
    std::cout << "Server thread finished init" << std::endl;
    
    std::cout << "A" << std::endl;
    // Launch the two kernels
    completion_queue_thread_func<<<1,1,0,completionQueueStream>>>(h_cq_p);
    std::cout << "B" << std::endl;
    submission_queue_thread_func<<<1,1,0,submissionQueueStream>>>(h_sq_p, NUM_ITERATIONS, 644);
    std::cout << "C" << std::endl;

    cudaStreamSynchronize(completionQueueStream);
    cudaStreamSynchronize(submissionQueueStream);

    // Destroy streams
    cudaStreamDestroy(submissionQueueStream);
    cudaStreamDestroy(completionQueueStream);

    // std::cout << "Before" << std::endl;
    // cudaDeviceSynchronize();
    // std::cout << "After" << std::endl;


    std::cout << "Calling PLIOPS_CloseDB!" << std::endl;       
    ret = PLIOPS_CloseDB(plio_handle);
    if (ret != 0) {
        printf("PLIOPS_CloseDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_CloseDB!" << std::endl;       

    std::cout << "Calling PLIOPS_DeleteDB!" << std::endl;       
    ret = PLIOPS_DeleteDB(identify, 0);
    if (ret != 0) {
        printf("PLIOPS_DeleteDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_DeleteDB!" <<std::endl;  


	cudaFreeHost(h_sq_p);
	cudaFreeHost(h_cq_p);
    return 0;
}