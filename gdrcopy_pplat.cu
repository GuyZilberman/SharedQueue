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
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime_api.h>
 #include <thread>

using namespace std;

#include "gdrapi.h"
#include "gdrcopy_common.hpp"

using namespace gdrcopy::test;

void pp_cpu_thread(gdr_mh_t mh, uint32_t *d_buf, uint32_t *h_buf, uint32_t num_iters){
        uint32_t i = 1;
        // Wait for pp_kernel to be ready before starting the time measurement.
        while (READ_ONCE(*h_buf) != i);
        LB();

        // Restart the timer for measurement.
        while (i < num_iters) {
            gdr_copy_to_mapping(mh, d_buf, &i, sizeof(d_buf));
            SB();

            ++i;

            while (READ_ONCE(*h_buf) != i);
            LB();
        }
}

__global__ void pp_kernel(uint32_t *d_buf, uint32_t *h_buf, uint32_t num_iters)
{
    uint32_t i = 1;
    WRITE_ONCE(*h_buf, i);
    __threadfence_block();
    while (i < num_iters) {
        while (READ_ONCE(*d_buf) != i) ;
        __threadfence_block();

        ++i;
        WRITE_ONCE(*h_buf, i);
        __threadfence_block();
    }
}

static int dev_id = 0;
static uint32_t num_iters = 5;

int main(int argc, char *argv[])
{
    uint32_t *d_buf = NULL;
    uint32_t *h_buf = NULL;

    CUdeviceptr d_buf_cuptr;
    CUdeviceptr h_buf_cuptr;

    gpu_mem_handle_t mhandle;

    // GUY: Initialize the CUDA driver API
    ASSERTDRV(cuInit(0));

    // GUY: Start of device selection stuff
    int n_devices = 0;
    ASSERTDRV(cuDeviceGetCount(&n_devices));

    CUdevice dev;
    for (int n=0; n<n_devices; ++n) {
        
        char dev_name[256];
        int dev_pci_domain_id;
        int dev_pci_bus_id;
        int dev_pci_device_id;

        ASSERTDRV(cuDeviceGet(&dev, n));
        ASSERTDRV(cuDeviceGetName(dev_name, sizeof(dev_name) / sizeof(dev_name[0]), dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));

        cout  << "GPU id:" << n << "; name: " << dev_name 
              << "; Bus id: "
              << std::hex 
              << std::setfill('0') << std::setw(4) << dev_pci_domain_id
              << ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id
              << ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id
              << std::dec
              << endl;
    }
    cout << "selecting device " << dev_id << endl;
    ASSERTDRV(cuDeviceGet(&dev, dev_id));

    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));

    // Check that the device supports GDR
    ASSERT_EQ(check_gdr_support(dev), true);
    // GUY: End of device selection stuff

    ASSERTDRV(gpu_mem_alloc(&mhandle, sizeof(*d_buf), true, true));
    d_buf_cuptr = mhandle.ptr;
    cout << "device ptr: 0x" << hex << d_buf_cuptr << dec << endl;

    // set d_buf_cuptr's value to 0
    ASSERTDRV(cuMemsetD8(d_buf_cuptr, 0, sizeof(*d_buf)));

    ASSERTDRV(cuMemHostAlloc((void **)&h_buf, sizeof(*h_buf), CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    ASSERT_NEQ(h_buf, (void*)0);
    ASSERTDRV(cuMemHostGetDevicePointer(&h_buf_cuptr, h_buf, 0));
    memset(h_buf, 0, sizeof(*h_buf));

    // called to open a handle to the GPUDirect RDMA driver
    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // Create a peer-to-peer mapping of the device memory buffer, returning an opaque handle.
        ASSERT_EQ(gdr_pin_buffer(g, d_buf_cuptr, sizeof(*d_buf), 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, sizeof(*d_buf)), 0);
        cout << "map_d_ptr: " << map_d_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        cout << "info.va: " << hex << info.va << dec << endl;
        cout << "info.mapped_size: " << info.mapped_size << endl;
        cout << "info.page_size: " << info.page_size << endl;
        cout << "info.mapped: " << info.mapped << endl;
        cout << "info.wc_mapping: " << info.wc_mapping << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = info.va - d_buf_cuptr;
        cout << "page offset: " << off << endl;

        d_buf = (uint32_t *)((uintptr_t)map_d_ptr + off);
        cout << "user-space pointer: " << d_buf << endl;

        cout << "CPU does gdr_copy_to_mapping and GPU writes back via cuMemHostAlloc'd buffer." << endl;
        cout << "Running " << num_iters << " iterations with data size " << sizeof(*d_buf) << " bytes." << endl;

        pp_kernel<<< 1, 1 >>>((uint32_t *)d_buf_cuptr, (uint32_t *)h_buf_cuptr, num_iters);

        // Catching any potential errors. CUDA_ERROR_NOT_READY means pp_kernel
        // is running. We expect to see this status instead of CUDA_SUCCESS
        // because pp_kernel must wait for signal from CPU, which occurs after
        // this line.
        ASSERT_EQ(cuStreamQuery(0), CUDA_ERROR_NOT_READY);

        // Launch a server thread
        std::thread server_thread(pp_cpu_thread, mh, d_buf, h_buf, num_iters);
        server_thread.detach();

        ASSERTDRV(cuStreamSynchronize(0));

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, sizeof(*d_buf)), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFreeHost(h_buf));
    ASSERTDRV(gpu_mem_free(&mhandle));

    return 0;
}
