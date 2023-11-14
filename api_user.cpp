#include <iostream>
#include <vector>
#include "/etc/pliops/store_lib_expo.h"

#define NO_OPTIONS 0


// TODO guy: check why I had to use export LD_LIBRARY_PATH=/etc/pliops:$LD_LIBRARY_PATH for it to work.
int main(){
    PLIOPS_IDENTIFY_t identify = 0; //TODO guy check if I need a better identifier
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options; //TODO guy check if I need other options
    db_open_options.createIfMissing = 1;
    int ret;
    PLIOPS_DB_t plio_handle;
    uint key = 0, read_val = 0, actual_object_size; //TODO guy make uint
    std::vector<uint> myVector = {1337, 322, 420, 30};

    std::cout << "Calling PLIOPS_OpenDB!" <<std::endl;       
    ret = PLIOPS_OpenDB(identify, &db_open_options, 0, &plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        exit(1);
    }
    std::cout << "Finished PLIOPS_OpenDB!" <<std::endl;       

    for (int i = 0 ; i < myVector.size() ; i++){
        std::cout << i << ": Calling PLIOPS_Put! Value: "  << myVector[i] << std::endl;
        ret = PLIOPS_Put(plio_handle, &i, sizeof(i), &myVector[i], sizeof(myVector[i]), NO_OPTIONS); //TODO guy look into options
        if (ret != 0) {
            printf("PLIOPS_Put Failed ret=%d\n", ret);
            exit(1);
        }
        std::cout << "Finished PLIOPS_Put!" << std::endl; 
    }

    for (int i = 0 ; i < myVector.size() ; i++){
        std::cout << "Calling PLIOPS_Get!" <<std::endl;
        ret = PLIOPS_Get(plio_handle, &i, sizeof(i), &read_val, sizeof(read_val), &actual_object_size); //TODO guy look into options
        if (ret != 0) {
            printf("PLIOPS_Get Failed ret=%d\n", ret);
            exit(1);
        }
        std::cout << "Finished PLIOPS_Get!" <<std::endl; 
        std::cout << i << ": Called PLIOPS_Get! Value: "  << read_val << std::endl;
    }

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


    return 0;
}