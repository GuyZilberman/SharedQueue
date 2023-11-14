#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"

int main() {
    int shm_fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(LockFreeQueue));
    LockFreeQueue *queue = static_cast<LockFreeQueue*>(mmap(0, sizeof(LockFreeQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));

    new (queue) LockFreeQueue();

    while (true) {
        int value;
        while (!queue->pop(value)); // Busy-wait for a value to be available
        std::cout << "Received: " << value << std::endl;
    }

    munmap(queue, sizeof(LockFreeQueue));
    close(shm_fd);
    shm_unlink(SHARED_MEMORY_NAME);
    return 0;
}
