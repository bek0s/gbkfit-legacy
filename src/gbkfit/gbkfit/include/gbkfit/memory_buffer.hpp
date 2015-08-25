#pragma once
#ifndef GBKFIT_MEMORY_BUFFER_HPP
#define GBKFIT_MEMORY_BUFFER_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


//!
//! \brief The memory_buffer class
//!
class memory_buffer
{

public:

    typedef typename std::size_t size_type;

private:

    size_type m_size;

public:

    memory_buffer(size_type size);

    virtual ~memory_buffer();

    size_type get_size(void) const;

    void read_data(void* dst) const;

    void write_data(const void* src);

    void copy_data(const memory_buffer* src);

    virtual void read_data(void* dst, size_type src_offset, size_type length) const = 0;

    virtual void write_data(const void* src, size_type dst_offset, size_type length) = 0;

    virtual void copy_data(const memory_buffer* src, size_type src_offset, size_type dst_offset, size_type length) = 0;

}; // class memory_buffer


} // namespace gbkfit

#endif // GBKFIT_MEMORY_BUFFER_HPP
