#include <atomic>
#include <iostream>

#define SUBMISSION_QUEUE_NAME "/submission_queue"
#define COMPLETION_QUEUE_NAME "/completion_queue"

const int QUEUE_SIZE = 3;

//https://chat.openai.com/c/1014224d-4eb5-45ae-9862-a60a3b162a88

template<typename T>
class LockFreeQueue {
private:
    std::atomic<int> head;
    std::atomic<int> tail;
    T data[QUEUE_SIZE];

public:
    LockFreeQueue() : head(0), tail(0) {}

    bool push(T val) {
        int currTail = tail.load(); //CS AFTER
        if ((currTail + 1) % QUEUE_SIZE == head.load()) {
            return false; // Queue full
        }
        data[currTail] = val;
        tail.store((currTail + 1) % QUEUE_SIZE);
        return true;
    }

    bool pop(T &val) {
        int currHead = head.load(); //CS AFTER?
        if (currHead == tail.load()) {
            return false; // Queue empty
        }
        val = data[currHead];
        head.store((currHead + 1) % QUEUE_SIZE);
        return true;
    }
};
