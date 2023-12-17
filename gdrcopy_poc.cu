#include <cuda/atomic>
#include "gdrcopy_common.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>

#include "gdrcopy_common.hpp"
#include "gdrapi.h"



using namespace gdrcopy::test;
using namespace std;
// manually tuned... //TODO guy figure this out
int num_write_iters = 10000;
int num_read_iters  = 100;
size_t _size = 128*1024; // Buffer size in bytes? Its maximal valid value is copy_offset+copy_size
size_t copy_size = 0; // Copy size in bytes?
size_t copy_offset_in_bytes = 0; // Copy offset in bytes
int dev_id = 0;


__global__
void GPU_thread(cuda::atomic<int>* flag) {
    printf("Hello from GPU_thread\n");
    for (int i = 0; i < 5; i++)
    {
        while(flag->load() != 0){}
        printf("GPU_thread: %d\n", i);
        flag->store(1);
    }
    printf("Bye from GPU_thread\n");
}


void run_test_no_test(CUdeviceptr d_A, size_t size)
{
    // GUY - MUST
    // 2. gdr_open_safe() is called to open a handle to the 
    // GPUDirect RDMA driver, enabling direct memory access capabilities.
    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled

        // GUY - MUST
        // 3.a. "Pin" the GPU buffer, which means marking it for direct DMA access. 
        // This step is essential for RDMA as it prepares the GPU memory for 
        // direct access by other devices.
        // Create a peer-to-peer mapping of the device memory buffer, returning an opaque handle.
        BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        // GUY - MUST
        // 3.b. Map the pinned buffer into the CPU address space, allowing the host to access 
        // this memory directly. This is a key step in setting up RDMA, as it provides a 
        // way for the CPU to directly read from and write to GPU memory.
        ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, size), 0);
        cout << "map_d_ptr: " << map_d_ptr << endl;

        gdr_info_t info;
        // GUY - MUST
        // 4. Retrieve information about the mapped GPU buffer, such as its 
        // virtual address, size, and whether it's been mapped with write-combining (WC) 
        // memory (which can affect performance).
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        /*
        cout << "info.va: " << hex << info.va << dec << endl;
        cout << "info.mapped_size: " << info.mapped_size << endl;
        cout << "info.page_size: " << info.page_size << endl;
        cout << "info.mapped: " << info.mapped << endl;
        cout << "info.wc_mapping: " << info.wc_mapping << endl;
        */

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer

        // GUY - MUST
        // GUY: the address obtained by gdr_map could be potentially aligned to the boundary 
        // of the page size before being mapped in user-space, so the pointer returned might be
        // affected by an offset. gdr_get_info can be used to calculate that offset, 
        // given by (the virtual address - d_A)
        int off = info.va - d_A;

        // GUY - MUST
        // GUY: This is the result
        uint32_t *buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
        cout << "user-space pointer:" << buf_ptr << endl;


        // ~~~Ptr deinit starts here~~~
        // GUY - MUST
        // unmapping buffer
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);

        // GUY - MUST
        // unpinning buffer
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
        // ~~~Ptr deinit ends here~~~
    } END_CHECK;

    // ~~~Session deinit starts here~~~
    // GUY - MUST
    // closing gdr driver
    ASSERT_EQ(gdr_close(g), 0);
    // ~~~Session deinit ends here~~~
}


void run_test(CUdeviceptr d_A, size_t size)
{
    uint32_t *init_buf = NULL;

    // 1. Allocating host memory. 
    // That will be used to initialize and compare data. 
    // This allocation method ensures that the memory is suitable for 
    // efficient DMA (Direct Memory Access) transfers.
    ASSERTDRV(cuMemAllocHost((void **)&init_buf, size));
    ASSERT_NEQ(init_buf, (void*)0);
    init_hbuf_walking_bit(init_buf, size);

    // GUY - MUST
    // 2. gdr_open_safe() is called to open a handle to the 
    // GPUDirect RDMA driver, enabling direct memory access capabilities.
    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled

        // GUY - MUST
        // 3.a. "Pin" the GPU buffer, which means marking it for direct DMA access. 
        // This step is essential for RDMA as it prepares the GPU memory for 
        // direct access by other devices.
        BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        // GUY - MUST
        // 3.b. Map the pinned buffer into the CPU address space, allowing the host to access 
        // this memory directly. This is a key step in setting up RDMA, as it provides a 
        // way for the CPU to directly read from and write to GPU memory.
        ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, size), 0);
        cout << "map_d_ptr: " << map_d_ptr << endl;

        gdr_info_t info;
        // GUY - MUST
        // 4. Retrieve information about the mapped GPU buffer, such as its 
        // virtual address, size, and whether it's been mapped with write-combining (WC) 
        // memory (which can affect performance).
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        /*
        cout << "info.va: " << hex << info.va << dec << endl;
        cout << "info.mapped_size: " << info.mapped_size << endl;
        cout << "info.page_size: " << info.page_size << endl;
        cout << "info.mapped: " << info.mapped << endl;
        cout << "info.wc_mapping: " << info.wc_mapping << endl;
        */

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer

        // GUY - MUST
        // GUY: the address obtained by gdr_map could be potentially aligned to the boundary 
        // of the page size before being mapped in user-space, so the pointer returned might be
        // affected by an offset. gdr_get_info can be used to calculate that offset, 
        // given by (the virtual address - d_A)
        int off = info.va - d_A;
        cout << "page offset: " << off << endl;

        // GUY - MUST
        // GUY: This is the result
        uint32_t *buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
        cout << "user-space pointer:" << buf_ptr << endl;

        // copy to GPU benchmark
        cout << "writing test, size=" << copy_size << " offset=" << copy_offset_in_bytes << " num_iters=" << num_write_iters << endl;
        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);
        for (int iter=0; iter<num_write_iters; ++iter)
            // 5.a. Writing data to GPU memory (gdr_copy_to_mapping()) 
            // and measuring the time taken for multiple iterations.
            gdr_copy_to_mapping(mh, buf_ptr + copy_offset_in_bytes/4, init_buf, copy_size);
        clock_gettime(MYCLOCK, &end);

        double woMBps;
        // 5.a. ctd. Calcualte the write BW in MB/s
        {
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "write BW: " << woMBps << "MB/s" << endl;
        }

        // 6. After the data transfers, there's a comparison between the original data 
        // and the data read back from the GPU to ensure the integrity of the data transfer.
        compare_buf(init_buf, buf_ptr + copy_offset_in_bytes/4, copy_size);

        // copy from GPU benchmark
        cout << "reading test, size=" << copy_size << " offset=" << copy_offset_in_bytes << " num_iters=" << num_read_iters << endl;
        clock_gettime(MYCLOCK, &beg);
        for (int iter=0; iter<num_read_iters; ++iter)
            // 5.b. Reading data back from GPU memory (gdr_copy_from_mapping()) 
            // and measuring the time for these operations.
            gdr_copy_from_mapping(mh, init_buf, buf_ptr + copy_offset_in_bytes/4, copy_size);
        clock_gettime(MYCLOCK, &end);

        // 5.b. ctd. Calcualte the write BW in MB/s
        double roMBps;
        {
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "read BW: " << roMBps << "MB/s" << endl;
        }

        // GUY - MUST
        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);

        // GUY - MUST
        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    // GUY - MUST
    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);
}

int main() {
    // GUY: Start of test related stuff - TODO GUY DELETE
    if (!copy_size)
        copy_size = _size;

    if (copy_offset_in_bytes % sizeof(uint32_t) != 0) {
        fprintf(stderr, "ERROR: offset must be multiple of 4 bytes\n");
        exit(EXIT_FAILURE);
    }

    if (copy_offset_in_bytes + copy_size > _size) {
        fprintf(stderr, "ERROR: offset + copy size run past the end of the buffer\n");
        exit(EXIT_FAILURE);
    }
    // GUY: End of test related stuff - TODO GUY DELETE

    size_t size = PAGE_ROUND_UP(_size, GPU_PAGE_SIZE);

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

        cout << "GPU id:" << n << "; name: " << dev_name 
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
    // GUY: End of device selection stuff

    // Check that the device supports GDR
    ASSERT_EQ(check_gdr_support(dev), true);

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(gpu_mem_alloc(&mhandle, size, true, true));
    d_A = mhandle.ptr;
    cout << "device ptr: " << hex << d_A << dec << endl;

    //run_test(d_A, size);
    run_test_no_test(d_A, size);

    ASSERTDRV(gpu_mem_free(&mhandle));

    ASSERTDRV(cuDevicePrimaryCtxRelease(dev));
    return 0;
}