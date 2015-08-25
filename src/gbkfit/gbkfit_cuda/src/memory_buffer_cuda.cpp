
#include "gbkfit/cuda/memory_buffer_cuda.hpp"
#include <cuda_runtime_api.h>

namespace gbkfit {
namespace cuda {


memory_buffer_cuda::memory_buffer_cuda(size_type size)
    : memory_buffer(size),
      m_data(nullptr)
{
}

memory_buffer_cuda::~memory_buffer_cuda()
{
}

void* memory_buffer_cuda::get_cuda_ptr(void)
{
    return m_data;
}

const void* memory_buffer_cuda::get_cuda_ptr(void) const
{
    return m_data;
}

void memory_buffer_cuda::read_data(void* dst, size_type src_offset, size_type length) const
{
    const std::uint8_t* src = m_data+src_offset;
    cudaMemcpy(dst,src,length,cudaMemcpyDefault);
}

void memory_buffer_cuda::write_data(const void* src, size_type dst_offset, size_type length)
{
    std::uint8_t* dst = m_data+dst_offset;
    cudaMemcpy(dst,src,length,cudaMemcpyDefault);
}

void memory_buffer_cuda::copy_data(const memory_buffer* src, size_type src_offset, size_type dst_offset, size_type length)
{
    if(const memory_buffer_cuda* src_buffer = dynamic_cast<const memory_buffer_cuda*>(src))
    {
        copy_data(src_buffer,src_offset,dst_offset,length);
    }
    else
    {
        // throw
    }
}

void memory_buffer_cuda::copy_data(const memory_buffer_cuda* src, size_type src_offset, size_type dst_offset, size_type length)
{
    std::uint8_t* dst_data = m_data+dst_offset;
    src->read_data(dst_data,src_offset,length);
}

memory_buffer_cuda_device::memory_buffer_cuda_device(size_type size)
    : memory_buffer_cuda(size)
{
    cudaMalloc((void**)&m_data,size);
}

memory_buffer_cuda_device::~memory_buffer_cuda_device()
{
    cudaFree(m_data);
}

memory_buffer_cuda_pinned::memory_buffer_cuda_pinned(size_type size)
    : memory_buffer_cuda(size)
{
    unsigned int flags = cudaHostAllocDefault;
    cudaHostAlloc((void**)&m_data,size,flags);
}

memory_buffer_cuda_pinned::~memory_buffer_cuda_pinned()
{
    cudaFreeHost(m_data);
}

memory_buffer_cuda_pinned_wc::memory_buffer_cuda_pinned_wc(size_type size)
    : memory_buffer_cuda(size)
{
    unsigned int flags = cudaHostAllocWriteCombined;
    cudaHostAlloc((void**)&m_data,size,flags);
}

memory_buffer_cuda_pinned_wc::~memory_buffer_cuda_pinned_wc()
{
    cudaFreeHost(m_data);
}


} // namespace cuda
} // namespace gbkfit
