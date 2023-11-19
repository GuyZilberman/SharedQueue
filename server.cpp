#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"
#include "/etc/pliops/store_lib_expo.h"
#include <cstring>

#define NO_OPTIONS 0

template<typename T>
struct SharedResources {
    LockFreeQueue<T> *submission_queue;
    LockFreeQueue<T> *completion_queue;
    int submission_shm_fd;
    int completion_shm_fd;
    PLIOPS_DB_t plio_handle;
    PLIOPS_IDENTIFY_t identify;
};

template<typename T>
int open_and_size_shm(const char* name) {
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) return -1; // Handle error appropriately

    if (ftruncate(fd, sizeof(T)) == -1) {
        close(fd); // Handle error appropriately
        return -1;
    }

    return fd;
}

template<typename T>
T* map_and_init_queue(int shm_fd) {
    T* queue = static_cast<T*>(mmap(0, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
    if (queue == MAP_FAILED) return nullptr; // Handle error appropriately

    new (queue) T(); // Placement new to initialize the queue
    return queue;
}

template<typename T>
bool setup_queue(SharedResources<T> &resources, const char* name) {
    int *shm_fd_ptr;
    LockFreeQueue<T>** queue_ptr;

    if (strcmp(name, SUBMISSION_QUEUE_NAME) == 0) {
        shm_fd_ptr = &resources.submission_shm_fd;
        queue_ptr = &resources.submission_queue;
    } else if (strcmp(name, COMPLETION_QUEUE_NAME) == 0) {
        shm_fd_ptr = &resources.completion_shm_fd;
        queue_ptr = &resources.completion_queue;
    } else {
        std::cerr << "Wrong name" << std::endl; // Improved error message
        return false;
    }
    
    *shm_fd_ptr = open_and_size_shm<LockFreeQueue<T>>(name);

    if (*shm_fd_ptr == -1) {
        std::cerr << "Failed initializing shared memory" << std::endl;
        return false;
    }

    *queue_ptr = map_and_init_queue<LockFreeQueue<T>>(*shm_fd_ptr);

    if (!(*queue_ptr)) {
        std::cerr << "Queue initialization failed" << std::endl;
        return false;
    }
    return true;
}

template<typename T>
bool init(SharedResources<T> &resources){
    setup_queue<T>(resources, SUBMISSION_QUEUE_NAME);
    setup_queue<T>(resources, COMPLETION_QUEUE_NAME);

    // Start of storelib init
    resources.identify = 0; //TODO guy check if I need a better identifier
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options; //TODO guy check what each flag in the option does
    db_open_options.createIfMissing = 1;
    db_open_options.tailSizeInBytes = 0;
    // TODO guy ask ido: db_open_options.errorIfExists = ???

    std::cout << "Calling PLIOPS_OpenDB!" << std::endl;       
    int ret = PLIOPS_OpenDB(resources.identify, &db_open_options, 0, &resources.plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_OpenDB!" << std::endl;       
    return true;
    // End of storelib init
    }

template<typename T>
bool deinit(SharedResources<T> &resources){
    // Start of storelib deinit
    std::cout << "Calling PLIOPS_CloseDB!" << std::endl;       
    int ret = PLIOPS_CloseDB(resources.plio_handle);
    if (ret != 0) {
        printf("PLIOPS_CloseDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_CloseDB!" << std::endl;       

    std::cout << "Calling PLIOPS_DeleteDB!" << std::endl;       
    ret = PLIOPS_DeleteDB(resources.identify, 0);
    if (ret != 0) {
        printf("PLIOPS_DeleteDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_DeleteDB!" <<std::endl;  
    // End of storelib deinit

    munmap(resources.submission_queue, sizeof(LockFreeQueue<T>)); //TODO guy
    close(resources.submission_shm_fd); //TODO guy
    shm_unlink(SUBMISSION_QUEUE_NAME); //TODO guy
    return true;
}

template<typename T>
bool process_requests(SharedResources<T> &resources){
    uint key = 0, read_val = 0, actual_object_size = 0, idx = 0;
    int ret = 0, value = 0;
    while (value != -1) {
        while (!resources.submission_queue->pop(value)); // Busy-wait for a value to be available
        std::cout << "Received: " << value << std::endl;

        std::cout << idx << ": Calling PLIOPS_Put! Value: "  << value << std::endl;
        ret = PLIOPS_Put(resources.plio_handle, &idx, sizeof(idx), &value, sizeof(value), NO_OPTIONS); //TODO guy look into options
        if (ret != 0) {
            printf("PLIOPS_Put Failed ret=%d\n", ret);
            return false;
        }
        std::cout << "Finished PLIOPS_Put!" << std::endl; 


        std::cout << "Calling PLIOPS_Get!" << std::endl;
        ret = PLIOPS_Get(resources.plio_handle, &idx, sizeof(idx), &read_val, sizeof(read_val), &actual_object_size);
        if (ret != 0) {
            printf("PLIOPS_Get Failed ret=%d\n", ret);
            return false;
        }
        std::cout << "Finished PLIOPS_Get!" << std::endl; 
        std::cout << idx << ": Called PLIOPS_Get! Value: "  << read_val << std::endl;

        while (!resources.completion_queue->push(read_val)); // Busy-wait until the value is pushed successfully
        std::cout << "Server sent confirmation for the following value: " << read_val << std::endl;


        idx = idx + 1;
    }
    return true;
}

int main() {
    SharedResources<int> resources;

    if (!init<int>(resources)) {
        std::cout << "Initialization failed. Exiting." << std::endl;       
        return 1;
    }

    if (!process_requests(resources)) {
        std::cout << "Request processing failed. Exiting." << std::endl;       
        return 1;
    }

    if (!deinit<int>(resources)) {
        std::cout << "Deinitialization failed. Exiting." << std::endl;       
        return 1;
    }
    
    return 0;
}
