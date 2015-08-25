#pragma once
#ifndef GBKFIT_FFTW3_MEMORY_BUFFER_FFTW3_HPP
#define GBKFIT_FFTW3_MEMORY_BUFFER_FFTW3_HPP

#include "gbkfit/memory_buffer_host.hpp"

namespace gbkfit {
namespace fftw3 {


//!
//! \brief The memory_buffer_fftw3 class
//!
class memory_buffer_fftw3 : public memory_buffer_host
{
public:
    memory_buffer_fftw3(size_type size);
    ~memory_buffer_fftw3();
}; // class memory_buffer_fftw3


} // namespace fftw3
} // namespace gbkfit

#endif // GBKFIT_FFTW3_MEMORY_BUFFER_FFTW3_HPP
