#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.cuh"
#include <thread>
#include <random>
#include <vector>
#include <algorithm>

#define NUM_ITERATIONS 10
#define READ_START_ID NUM_ITERATIONS
#define READ_END_ID 2*NUM_ITERATIONS-1



void generateRandomInts(int* arr, size_t size, int idx) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());


    // Set the first value in the vector to idx
    arr[0] = idx;

    // Set the rest to random values
    for (size_t i = 1; i < size; ++i) {
        arr[i] = dis(gen);
    }
}


void completion_queue_thread_func() {
    int cq_shm_fd = shm_open(COMPLETION_QUEUE_NAME, O_RDWR, 0666);
    LockFreeQueue<ResponseMessage> *completion_queue = static_cast<LockFreeQueue<ResponseMessage>*>(mmap(0, sizeof(LockFreeQueue<ResponseMessage>), PROT_READ | PROT_WRITE, MAP_SHARED, cq_shm_fd, 0));
    AnswerType answer = AnswerType::NONE;
    int wrong_answers = 0; //TODO guy - WRITE-READ test

    while (answer != AnswerType::EXIT) {
        ResponseMessage res_msg;
        while (!completion_queue->pop(res_msg)); // Busy-wait for a command to be available
        // TODO guy check about this: Optional: backoff strategy to reduce CPU usage
        answer = res_msg.answer;
        std::cout << "Client: Received from completion queue: " << (int)answer << std::endl;

        if (res_msg.request_id >= READ_START_ID && res_msg.request_id <= READ_END_ID){ //TODO guy - WRITE-READ test
            if (res_msg.data[0] != res_msg.request_id - NUM_ITERATIONS){
                std::cout << res_msg.request_id << " Mistake found!" << std::endl;
                wrong_answers++;
            }
        }

    }
    std::cout << "wrong_answers: " << wrong_answers << std::endl;
    munmap(completion_queue, sizeof(LockFreeQueue<ResponseMessage>)); //TODO guy change to <ResponseMessage>
    close(cq_shm_fd);
}


void submission_queue_thread_func(int num_iterations, int value_size) {
    int sq_shm_fd = shm_open(SUBMISSION_QUEUE_NAME, O_RDWR, 0666);
    LockFreeQueue<RequestMessage> *submission_queue = static_cast<LockFreeQueue<RequestMessage>*>(mmap(0, sizeof(LockFreeQueue<RequestMessage>), PROT_READ | PROT_WRITE, MAP_SHARED, sq_shm_fd, 0));
    uint idx = 0, request_id = 0;

    
    // Send 10000 Write requests
    while (idx < num_iterations){
        RequestMessage req_msg;
        req_msg.cmd = CommandType::WRITE;
        req_msg.request_id = request_id++;
        generateRandomInts(req_msg.data, 256, idx);
        req_msg.key = idx++;

        while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully
    }

    // Send 10000 Read requests
    idx = 0;
    while (idx < num_iterations){
        RequestMessage req_msg;
        req_msg.cmd = CommandType::READ;
        req_msg.request_id = request_id++;
        req_msg.key = idx++;

        while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully
    }

    // Send exit request
    RequestMessage req_msg;
    req_msg.cmd = CommandType::EXIT;
    req_msg.request_id = request_id;

    while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully

    munmap(submission_queue, sizeof(LockFreeQueue<RequestMessage>));
    close(sq_shm_fd);
}

int main() {
    std::thread submission_queue_thread(submission_queue_thread_func, NUM_ITERATIONS, 644);
    std::thread completion_queue_thread(completion_queue_thread_func);
    submission_queue_thread.join();
    completion_queue_thread.join();
    return 0;
}
