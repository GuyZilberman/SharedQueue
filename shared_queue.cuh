#include <atomic>
#include <iostream>
#include <vector>
#include <cuda/atomic>
#include "gdrapi.h"

#define SUBMISSION_QUEUE_NAME "/submission_queue"
#define COMPLETION_QUEUE_NAME "/completion_queue"
#define DATA_ARR_SIZE 253 //TODO GUY 253

const int QUEUE_SIZE = 3;

#include <time.h>
#include <stdio.h>

#define GET_TIME(function_call, clockRatekHz, message) do { \
    clock_t start1 = clock(); \
    function_call; \
    clock_t end1 = clock(); \
    clock_t elapsedCycles1 = end1 - start1; \
    float timeMilliseconds1 = (float)elapsedCycles1 / (clockRatekHz); \
    printf("%s: %.2f milliseconds\n", message, timeMilliseconds1); \
} while (0)

#define GPU_CLK_RATE_KHZ 1410000 //GPU Clock Rate: 1410000.00 kHz

//https://chat.openai.com/c/1014224d-4eb5-45ae-9862-a60a3b162a88

enum class CommandType {
    NONE = -2,
    EXIT,
    WRITE,
    READ
};

enum class AnswerType {
    NONE = -2,
    EXIT,
    SUCCESS,
    FAIL
};

struct RequestMessage {
	uint request_id;
	CommandType cmd;
	uint key;
	int data[DATA_ARR_SIZE]; 
};

struct ResponseMessage {
	uint request_id;
	AnswerType answer;
    int error;
	int data[DATA_ARR_SIZE];
};


class HostAllocatedSubmissionQueue{
private:
    cuda::atomic<int> head;
    cuda::atomic<int> tail;
    RequestMessage data[QUEUE_SIZE];

public:
    HostAllocatedSubmissionQueue() : head(0), tail(0) {}

    __device__ 
    bool push(const RequestMessage &msg) {
        int currTail = tail.load(cuda::memory_order_relaxed); // TODO guy CS AFTER
        if ((currTail + 1) % QUEUE_SIZE == head.load(cuda::memory_order_acquire)) {
            return false; // Queue full
        }
        //data[currTail] = msg;
        GET_TIME(data[currTail] = msg, GPU_CLK_RATE_KHZ, "HostAllocatedSubmissionQueue - push");
        tail.store((currTail + 1) % QUEUE_SIZE, cuda::memory_order_release);
        return true;
    }

    __host__ 
    bool pop(RequestMessage &msg) {
        int currHead = head.load(cuda::memory_order_relaxed); //TODO guy CS AFTER?
        if (currHead == tail.load(cuda::memory_order_acquire)) {
            return false; // Queue empty
        }
        //msg = data[currHead];
        GET_TIME(msg = data[currHead], CLOCKS_PER_SEC, "HostAllocatedSubmissionQueue - pop");
        head.store((currHead + 1) % QUEUE_SIZE, cuda::memory_order_release);
        return true;
    }
};

class DeviceAllocatedCompletionQueue{
private:
    cuda::atomic<int> head;
    cuda::atomic<int> tail;
    ResponseMessage data[QUEUE_SIZE];
    gdr_mh_t mh;

public:
    DeviceAllocatedCompletionQueue(gdr_mh_t &mh) : head(0), tail(0), mh(mh) {
    }

    __host__
    bool push(const ResponseMessage &msg) {
        int currTail = tail.load(cuda::memory_order_relaxed); // TODO guy CS AFTER
        if ((currTail + 1) % QUEUE_SIZE == head.load(cuda::memory_order_acquire)) {
            return false; // Queue full
        }
        //data[currTail] = msg;
        //GET_TIME(data[currTail] = msg, CLOCKS_PER_SEC, "DeviceAllocatedCompletionQueue - push");
        //gdr_copy_to_mapping(mh, &(data[currTail]), &msg, sizeof(msg));
        GET_TIME(gdr_copy_to_mapping(mh, &(data[currTail]), &msg, sizeof(msg)), CLOCKS_PER_SEC, "DeviceAllocatedCompletionQueue - push");

        tail.store((currTail + 1) % QUEUE_SIZE, cuda::memory_order_release);
        return true;
    }

    __device__ 
    bool pop(ResponseMessage &msg) {
        int currHead = head.load(cuda::memory_order_relaxed); //TODO guy CS AFTER?
        if (currHead == tail.load(cuda::memory_order_acquire)) {
            return false; // Queue empty
        }
        GET_TIME(msg = data[currHead], GPU_CLK_RATE_KHZ, "DeviceAllocatedCompletionQueue - pop");
        //msg = data[currHead];
        head.store((currHead + 1) % QUEUE_SIZE, cuda::memory_order_release);
        return true;
    }
};
