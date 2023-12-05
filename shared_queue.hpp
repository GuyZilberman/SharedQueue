#include <atomic>
#include <iostream>
#include <vector>

#define SUBMISSION_QUEUE_NAME "/submission_queue"
#define COMPLETION_QUEUE_NAME "/completion_queue"

const int QUEUE_SIZE = 10000000;

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
    std::atomic<int> head;
    std::atomic<int> tail;
    T data[QUEUE_SIZE];

public:
    LockFreeQueue() : head(0), tail(0) {}

    bool push(T val) {
        int currTail = tail.load(); // TODO guy CS AFTER
        if ((currTail + 1) % QUEUE_SIZE == head.load()) {
            return false; // Queue full
        }
        data[currTail] = val;
        tail.store((currTail + 1) % QUEUE_SIZE);
        return true;
    }

    bool pop(T &val) {
        int currHead = head.load(); //TODO guy CS AFTER?
        if (currHead == tail.load()) {
            return false; // Queue empty
        }
        val = data[currHead];
        head.store((currHead + 1) % QUEUE_SIZE);
        return true;
    }
};
