
#include "gbkfit/fftw3/memory_buffer_fftw3.hpp"
#include <fftw3.h>

namespace gbkfit {
namespace fftw3 {


memory_buffer_fftw3::memory_buffer_fftw3(size_type size)
    : memory_buffer_host(size)
{
    m_data = reinterpret_cast<std::uint8_t*>(fftwf_malloc(size));
}

memory_buffer_fftw3::~memory_buffer_fftw3(void)
{
    fftwf_free(m_data);
}


} // namespace fftw3
} // namespace gbkfit
