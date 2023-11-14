#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "shared_queue.hpp"
#include "/etc/pliops/store_lib_expo.h"

#define NO_OPTIONS 0

bool init(){
    //TODO move stuff here
}

bool deinit(){
    //TODO move stuff here
}

int main() {
    int shm_fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666); //TODO guy
    ftruncate(shm_fd, sizeof(LockFreeQueue)); //TODO guy
    LockFreeQueue *queue = static_cast<LockFreeQueue*>(mmap(0, sizeof(LockFreeQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0)); //TODO guy
    new (queue) LockFreeQueue();

    // Start of storelib init
    PLIOPS_IDENTIFY_t identify = 0; //TODO guy check if I need a better identifier
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options; //TODO guy check if I need other options
    db_open_options.createIfMissing = 1;
    int ret;
    PLIOPS_DB_t plio_handle;
    uint key = 0, read_val = 0, actual_object_size;

    std::cout << "Calling PLIOPS_OpenDB!" <<std::endl;       
    ret = PLIOPS_OpenDB(identify, &db_open_options, 0, &plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        exit(1);
    }
    std::cout << "Finished PLIOPS_OpenDB!" <<std::endl;       
    // End of storelib init

    int value = 0, idx = 0;
    while (value != -1) {
        while (!queue->pop(value)); // Busy-wait for a value to be available
        std::cout << "Received: " << value << std::endl;

        std::cout << idx << ": Calling PLIOPS_Put! Value: "  << value << std::endl;
        ret = PLIOPS_Put(plio_handle, &idx, sizeof(idx), &value, sizeof(value), NO_OPTIONS); //TODO guy look into options
        if (ret != 0) {
            printf("PLIOPS_Put Failed ret=%d\n", ret);
            exit(1);
        }
        std::cout << "Finished PLIOPS_Put!" << std::endl; 


        std::cout << "Calling PLIOPS_Get!" <<std::endl;
        ret = PLIOPS_Get(plio_handle, &idx, sizeof(idx), &read_val, sizeof(read_val), &actual_object_size);
        if (ret != 0) {
            printf("PLIOPS_Get Failed ret=%d\n", ret);
            exit(1);
        }
        std::cout << "Finished PLIOPS_Get!" <<std::endl; 
        std::cout << idx << ": Called PLIOPS_Get! Value: "  << read_val << std::endl;

        idx = (idx + 1) % QUEUE_SIZE;
    }

    // Start of storelib deinit
    std::cout << "Calling PLIOPS_CloseDB!" <<std::endl;       
    ret = PLIOPS_CloseDB(plio_handle);
    if (ret != 0) {
        printf("PLIOPS_CloseDB Failed ret=%d\n", ret);
        exit(1);
    }
    std::cout << "Finished PLIOPS_CloseDB!" <<std::endl;       

    std::cout << "Calling PLIOPS_DeleteDB!" <<std::endl;       
    ret = PLIOPS_DeleteDB(identify, 0);
    if (ret != 0) {
        printf("PLIOPS_DeleteDB Failed ret=%d\n", ret);
        exit(1);
    }
    std::cout << "Finished PLIOPS_DeleteDB!" <<std::endl;  
    // End of storelib deinit

    munmap(queue, sizeof(LockFreeQueue)); //TODO guy
    close(shm_fd); //TODO guy
    shm_unlink(SHARED_MEMORY_NAME); //TODO guy
    return 0;
}
