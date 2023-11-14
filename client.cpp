#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"


int main() {
    int shm_fd = shm_open(SHARED_MEMORY_NAME, O_RDWR, 0666);
    LockFreeQueue *queue = static_cast<LockFreeQueue*>(mmap(0, sizeof(LockFreeQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));

    while (true) {
        int value;
        std::cout << "Enter a number: ";
        std::cin >> value;        
        while (!queue->push(value)); // Busy-wait until the value is pushed successfully
    }

    munmap(queue, sizeof(LockFreeQueue));
    close(shm_fd);
    return 0;
}
