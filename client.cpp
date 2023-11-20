#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"
#include <thread>


void pop_from_completion_queue() {
    int cq_shm_fd = shm_open(COMPLETION_QUEUE_NAME, O_RDWR, 0666);
    LockFreeQueue<ResponseMessage> *completion_queue = static_cast<LockFreeQueue<ResponseMessage>*>(mmap(0, sizeof(LockFreeQueue<ResponseMessage>), PROT_READ | PROT_WRITE, MAP_SHARED, cq_shm_fd, 0));
    int answer = -2;

    while (answer != -1) {
        ResponseMessage res_msg;
        while (!completion_queue->pop(res_msg)); // Busy-wait for a command to be available
        // TODO guy check about this: Optional: backoff strategy to reduce CPU usage
        answer = res_msg.answer;
        std::cout << "Client: Received from completion queue: " << answer << std::endl;

    }
    munmap(completion_queue, sizeof(LockFreeQueue<int>));
    close(cq_shm_fd);
}


void push_to_submission_queue() {
    int sq_shm_fd = shm_open(SUBMISSION_QUEUE_NAME, O_RDWR, 0666);
    LockFreeQueue<RequestMessage> *submission_queue = static_cast<LockFreeQueue<RequestMessage>*>(mmap(0, sizeof(LockFreeQueue<RequestMessage>), PROT_READ | PROT_WRITE, MAP_SHARED, sq_shm_fd, 0));
    int command;
    uint idx = 0;

    while (command != -1) {
        RequestMessage req_msg;
        command = -2;

        while (command != -1 && command != 0 && command != 1){
            std::cout << "Enter a command: (W = 0, R = 1, Exit = -1)" << std::endl;
            std::cin >> command;
            if (command != -1 && command != 0 && command != 1)
                std::cout << "Unauthorized command. ";
        }
        req_msg.cmd = static_cast<CommandType>(command);
        req_msg.request_id = idx;

        if (req_msg.cmd != EXIT)
        {
            if (req_msg.cmd == WRITE){
                std::cout << "Enter a value to store: " << std::endl;
                std::cin >> req_msg.data;
            }
            std::cout << "Enter a key: " << std::endl;
            std::cin >> req_msg.key;
        }

        while (!submission_queue->push(req_msg)); // Busy-wait until the value is pushed successfully
        idx++;
    }

    munmap(submission_queue, sizeof(LockFreeQueue<RequestMessage>));
    close(sq_shm_fd);
}

int main() {
    std::thread submission_queue_thread(push_to_submission_queue);
    std::thread completion_queue_thread(pop_from_completion_queue);
    submission_queue_thread.join();
    completion_queue_thread.join();
    return 0;
}
