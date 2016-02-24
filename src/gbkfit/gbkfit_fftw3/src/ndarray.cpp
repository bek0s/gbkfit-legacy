
#include "gbkfit/fftw3/ndarray.hpp"

#include <fftw3.h>

namespace gbkfit {
namespace fftw3 {

NDArray::NDArray(const NDShape& shape)
    : gbkfit::NDArray(shape)
{
    m_data = reinterpret_cast<pointer>(fftwf_malloc(get_size_in_bytes()));
}

NDArray::~NDArray()
{
    fftwf_free(m_data);
}

NDArray::pointer NDArray::get_host_ptr(void)
{
    return m_data;
}

NDArray::const_pointer NDArray::get_host_ptr(void) const
{
    return m_data;
}

void NDArray::read_data(pointer dst) const
{
    std::copy_n(m_data, get_size(), dst);
}

void NDArray::write_data(const_pointer src)
{
    std::copy_n(src, get_size(), m_data);
}

void NDArray::write_data(const gbkfit::NDArray* src)
{
    src->read_data(m_data);
}

NDArray::pointer NDArray::map(void)
{
    return m_data;
}

void NDArray::unmap(void)
{
}

} // namespace fftw3
} // namespace gbkfit
