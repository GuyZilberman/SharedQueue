#include <atomic>
#include <iostream>
#include <vector>
#include <cuda/atomic>

#define SUBMISSION_QUEUE_NAME "/submission_queue"
#define COMPLETION_QUEUE_NAME "/completion_queue"

const int QUEUE_SIZE = 3;

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
	int data[256];
};

struct ResponseMessage {
	uint request_id;
	AnswerType answer;
    int error;
	int data[256];
};

template<typename T>
class LockFreeQueue {
private:
    cuda::atomic<int> head;
    cuda::atomic<int> tail;
    T data[QUEUE_SIZE];

public:
    LockFreeQueue() : head(0), tail(0) {}

    __host__ __device__ 
    bool push(T val) {
        int currTail = tail.load(cuda::memory_order_relaxed); // TODO guy CS AFTER
        if ((currTail + 1) % QUEUE_SIZE == head.load(cuda::memory_order_acquire)) {
            return false; // Queue full
        }
        data[currTail] = val;
        tail.store((currTail + 1) % QUEUE_SIZE, cuda::memory_order_release);
        return true;
    }

    __host__ __device__ 
    bool pop(T &val) {
        int currHead = head.load(cuda::memory_order_relaxed); //TODO guy CS AFTER?
        if (currHead == tail.load(cuda::memory_order_acquire)) {
            return false; // Queue empty
        }
        val = data[currHead];
        head.store((currHead + 1) % QUEUE_SIZE, cuda::memory_order_release);
        return true;
    }
};
