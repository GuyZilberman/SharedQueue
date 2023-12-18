/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <getopt.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thread>

using namespace std;

#include "gdrapi.h"
#include "gdrcopy_common.hpp"
#include <cuda/atomic>
#include "gdr_gpu_memalloc.cuh"

using namespace gdrcopy::test;

class DummyClass{
private:
    int age;
public:
    cuda::atomic<int> id;
    int getAge(){
        return age;
    }
    void setAge(int age){
        this->age = age;
    }
};

void pp_cpu_thread(gdr_mh_t mh, DummyClass *d_buf, uint32_t num_iters){ // TODO guy !:!
        uint32_t i = 1, wval = 1, rval = 0;
        //LB(); //TODO guy !:!

        while (i < num_iters) {
            //gdr_copy_to_mapping(mh, &(d_buf->id), &wval, sizeof(&(d_buf->id))); // TODO guy !:!
            //SB(); //TODO guy !:!
            d_buf->id.store(wval);
            ++i;

            //while (READ_ONCE(d_buf->id) != rval); // TODO guy !:!
            while (d_buf->id.load() != rval); // TODO guy !:!
            //LB(); //TODO guy !:!
            printf("CPU: pp_cpu_thread: Finished iteration %d\n", i);
        }
}

__global__ 
void pp_kernel(DummyClass *d_buf, uint32_t num_iters) // TODO guy !:!
{
    uint32_t i = 1, rval = 1, wval = 0;;
    //__threadfence_block(); //TODO guy !:!
    while (i < num_iters) {
        //while (READ_ONCE(d_buf->id) != rval) ; //TODO guy !:!
        while (d_buf->id.load() != rval);
        //__threadfence_block(); //TODO guy !:!

        ++i;
        //WRITE_ONCE(d_buf->id, wval); //TODO guy !:!
        d_buf->id.store(wval);
        //__threadfence_block(); //TODO guy !:!
        printf("GPU: pp_kernel: Finished iteration %d\n", i);
    }
}


int main(int argc, char *argv[])
{
    DummyClass *d_buf = NULL;
    CUdeviceptr d_buf_cuptr;
    GPUMemoryManager *gpu_mm = new GPUMemoryManager();
    cudaGPUMemAlloc<DummyClass>(gpu_mm, &d_buf, d_buf_cuptr);

    BEGIN_CHECK {
        uint32_t num_iters = 5;
        pp_kernel<<<1, 1>>>((DummyClass *)d_buf_cuptr, num_iters);

        // CUDA_ERROR_NOT_READY means pp_kernel is running. We expect to see this
        // status instead of CUDA_SUCCESS because pp_kernel must wait for signal
        // from CPU, which occurs after this line.

        ASSERT_EQ(cuStreamQuery(0), CUDA_ERROR_NOT_READY);

        // Launch a server thread
        std::thread server_thread(pp_cpu_thread, gpu_mm->mh, d_buf, num_iters);
        server_thread.detach();

        ASSERTDRV(cuStreamSynchronize(0));
    } END_CHECK;

    cudaGPUMemFree<DummyClass>(gpu_mm);
    delete(gpu_mm);
    return 0;
}
