#pragma once
#ifndef GBKFIT_MEMORY_BUFFER_HOST_HPP
#define GBKFIT_MEMORY_BUFFER_HOST_HPP

#include "gbkfit/memory_buffer.hpp"

namespace gbkfit {


//!
//! \brief The memory_buffer_host class
//!
class memory_buffer_host : public memory_buffer
{

protected:

    std::uint8_t* m_data;

public:

    ~memory_buffer_host();

    void* get_host_ptr(void);

    const void* get_host_ptr(void) const;

    void read_data(void* dst, size_type src_offset, size_type length) const final;

    void write_data(const void* src, size_type dst_offset, size_type length) final;

    void copy_data(const memory_buffer* src, size_type src_offset, size_type dst_offset, size_type length) final;

    void copy_data(const memory_buffer_host* src, size_type src_offset, size_type dst_offset, size_type length);

protected:

    memory_buffer_host(size_type size);

}; // class memory_buffer_host


} // namespace gbkfit

#endif // GBKFIT_MEMORY_BUFFER_HOST_HPP
