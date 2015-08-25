#pragma once
#ifndef GBKFIT_CUDA_MEMORY_BUFFER_CUDA_HPP
#define GBKFIT_CUDA_MEMORY_BUFFER_CUDA_HPP

#include "gbkfit/memory_buffer.hpp"

namespace gbkfit {
namespace cuda {


//!
//! \brief The memory_buffer_cuda class
//!
class memory_buffer_cuda : public memory_buffer
{

protected:

    std::uint8_t* m_data;

public:

    ~memory_buffer_cuda();

    void* get_cuda_ptr(void);

    const void* get_cuda_ptr(void) const;

    void read_data(void* dst, size_type src_offset, size_type length) const final;

    void write_data(const void* src, size_type dst_offset, size_type length) final;

    void copy_data(const memory_buffer* src, size_type src_offset, size_type dst_offset, size_type length) final;

    void copy_data(const memory_buffer_cuda* src, size_type src_offset, size_type dst_offset, size_type length);

protected:

    memory_buffer_cuda(size_type size);

}; // class memory_buffer_cuda


//!
//! \brief The memory_buffer_cuda_device class
//!
class memory_buffer_cuda_device : public memory_buffer_cuda
{
public:
    memory_buffer_cuda_device(size_type size);
    ~memory_buffer_cuda_device();
}; // class memory_buffer_cuda_device


//!
//! \brief The memory_buffer_cuda_pinned class
//!
class memory_buffer_cuda_pinned : public memory_buffer_cuda
{
public:
    memory_buffer_cuda_pinned(size_type size);
    ~memory_buffer_cuda_pinned();
}; // class memory_buffer_cuda_pinned


//!
//! \brief The memory_buffer_cuda_pinned_wc class
//!
class memory_buffer_cuda_pinned_wc : public memory_buffer_cuda
{
public:
    memory_buffer_cuda_pinned_wc(size_type size);
    ~memory_buffer_cuda_pinned_wc();
}; // class memory_buffer_cuda_pinned_wc


} // namespace cuda
} // namespace gbkfit

#endif // GBKFIT_CUDA_MEMORY_BUFFER_CUDA_HPP
