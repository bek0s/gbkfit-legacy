#pragma once
#ifndef GBKFIT_MEMORY_BUFFER_MALLOC_HPP
#define GBKFIT_MEMORY_BUFFER_MALLOC_HPP

#include "gbkfit/memory_buffer_host.hpp"

namespace gbkfit {


//!
//! \brief The memory_buffer_malloc class
//!
class memory_buffer_malloc : public memory_buffer_host
{
public:
    memory_buffer_malloc(size_type size);
    ~memory_buffer_malloc();
}; // class memory_buffer_malloc


} // namespace gbkfit

#endif // GBKFIT_MEMORY_BUFFER_MALLOC_HPP
