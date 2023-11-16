#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"
#include "/etc/pliops/store_lib_expo.h"

#define NO_OPTIONS 0

struct SharedResources {
    LockFreeQueue *submission_queue;
    int shm_fd;
    PLIOPS_DB_t plio_handle;
    PLIOPS_IDENTIFY_t identify;
};

bool init(SharedResources &resources){
    resources.shm_fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666); //TODO guy
    ftruncate(resources.shm_fd, sizeof(LockFreeQueue)); //TODO guy
    resources.submission_queue = static_cast<LockFreeQueue*>(mmap(0, sizeof(LockFreeQueue), PROT_READ | PROT_WRITE, MAP_SHARED, resources.shm_fd, 0)); //TODO guy
    new (resources.submission_queue) LockFreeQueue();

    // Start of storelib init
    resources.identify = 0; //TODO guy check if I need a better identifier
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options; //TODO guy check if I need other options
    db_open_options.createIfMissing = 1;

    std::cout << "Calling PLIOPS_OpenDB!" <<std::endl;       
    int ret = PLIOPS_OpenDB(resources.identify, &db_open_options, 0, &resources.plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_OpenDB!" <<std::endl;       
    return true;
    // End of storelib init
    }

bool deinit(SharedResources &resources){
    // Start of storelib deinit
    std::cout << "Calling PLIOPS_CloseDB!" <<std::endl;       
    int ret = PLIOPS_CloseDB(resources.plio_handle);
    if (ret != 0) {
        printf("PLIOPS_CloseDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_CloseDB!" <<std::endl;       

    std::cout << "Calling PLIOPS_DeleteDB!" <<std::endl;       
    ret = PLIOPS_DeleteDB(resources.identify, 0);
    if (ret != 0) {
        printf("PLIOPS_DeleteDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_DeleteDB!" <<std::endl;  
    // End of storelib deinit

    munmap(resources.submission_queue, sizeof(LockFreeQueue)); //TODO guy
    close(resources.shm_fd); //TODO guy
    shm_unlink(SHARED_MEMORY_NAME); //TODO guy
    return true;
}

bool process_requests(SharedResources &resources){
    uint key = 0, read_val = 0, actual_object_size = 0;
    int ret = 0, value = 0, idx = 0;
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


        std::cout << "Calling PLIOPS_Get!" <<std::endl;
        ret = PLIOPS_Get(resources.plio_handle, &idx, sizeof(idx), &read_val, sizeof(read_val), &actual_object_size);
        if (ret != 0) {
            printf("PLIOPS_Get Failed ret=%d\n", ret);
            return false;
        }
        std::cout << "Finished PLIOPS_Get!" <<std::endl; 
        std::cout << idx << ": Called PLIOPS_Get! Value: "  << read_val << std::endl;

        idx = (idx + 1) % QUEUE_SIZE;
    }
}

int main() {
    SharedResources resources;

    if (!init(resources)) {
        std::cout << "Initialization failed. Exiting." <<std::endl;       
        return 1;
    }

    uint key = 0, read_val = 0, actual_object_size = 0;
    int ret = 0, value = 0, idx = 0;
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


        std::cout << "Calling PLIOPS_Get!" <<std::endl;
        ret = PLIOPS_Get(resources.plio_handle, &idx, sizeof(idx), &read_val, sizeof(read_val), &actual_object_size);
        if (ret != 0) {
            printf("PLIOPS_Get Failed ret=%d\n", ret);
            return false;
        }
        std::cout << "Finished PLIOPS_Get!" <<std::endl; 
        std::cout << idx << ": Called PLIOPS_Get! Value: "  << read_val << std::endl;

        idx = (idx + 1) % QUEUE_SIZE;
    }

    if (!deinit(resources)) {
        std::cout << "Deinitialization failed. Exiting." <<std::endl;       
        return 1;
    }
    
    return 0;
}
