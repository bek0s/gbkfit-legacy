
#include "gbkfit/fftw3/ndarray_host_fftw3.hpp"

#include <fftw3.h>

namespace gbkfit {
namespace fftw3 {

ndarray_host_fftw3::ndarray_host_fftw3(const NDShape& shape)
    : NDArrayHost(shape)
{
    std::size_t size = shape.get_dim_length_product()*sizeof(float);
    m_data = reinterpret_cast<pointer>(fftwf_malloc(size));
}

ndarray_host_fftw3::ndarray_host_fftw3(const NDShape& shape, const_pointer data)
    : ndarray_host_fftw3(shape)
{
    NDArrayHost::write_data(data);
}

} // namespace fftw3
} // namespace gbkfit
