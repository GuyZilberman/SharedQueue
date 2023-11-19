#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"
#include <thread>


void pop_from_completion_queue() {
    int cq_shm_fd = shm_open(COMPLETION_QUEUE_NAME, O_RDWR, 0666);
    LockFreeQueue<int> *completion_queue = static_cast<LockFreeQueue<int>*>(mmap(0, sizeof(LockFreeQueue<int>), PROT_READ | PROT_WRITE, MAP_SHARED, cq_shm_fd, 0));
    int value = 0;

    while (value != -1) {
        while (!completion_queue->pop(value)); // Busy-wait for a value to be available
        // TODO guy check about this: Optional: backoff strategy to reduce CPU usage
        std::cout << "Client: Received from completion queue: " << value << std::endl;

    }
    munmap(completion_queue, sizeof(LockFreeQueue<int>));
    close(cq_shm_fd);
}


void push_to_submission_queue() {
    int sq_shm_fd = shm_open(SUBMISSION_QUEUE_NAME, O_RDWR, 0666);
    LockFreeQueue<int> *submission_queue = static_cast<LockFreeQueue<int>*>(mmap(0, sizeof(LockFreeQueue<int>), PROT_READ | PROT_WRITE, MAP_SHARED, sq_shm_fd, 0));
    int value = 0;

    while (value != -1) {
        std::cout << "Enter a number: ";
        std::cin >> value;        
        while (!submission_queue->push(value)); // Busy-wait until the value is pushed successfully
    }

    munmap(submission_queue, sizeof(LockFreeQueue<int>));
    close(sq_shm_fd);
}

int main() {
    std::thread submission_queue_thread(push_to_submission_queue);
    std::thread completion_queue_thread(pop_from_completion_queue);
    submission_queue_thread.join();
    completion_queue_thread.join();
    return 0;
}
