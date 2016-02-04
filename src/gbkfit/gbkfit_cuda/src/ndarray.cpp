
#include "gbkfit/cuda/ndarray.hpp"

#include <cuda_runtime_api.h>

namespace gbkfit {
namespace cuda {

NDArray::NDArray(const NDShape& shape)
    : gbkfit::NDArray(shape)
    , m_data(nullptr)
{
}

NDArray::~NDArray()
{
}

gbkfit::NDArray::pointer NDArray::get_cuda_ptr(void)
{
    return m_data;
}

gbkfit::NDArray::const_pointer NDArray::get_cuda_ptr(void) const
{
    return m_data;
}

void NDArray::read_data(pointer dst) const
{
    cudaMemcpy(dst, m_data, get_size_in_bytes(), cudaMemcpyDefault);
}

void NDArray::write_data(const_pointer src)
{
    cudaMemcpy(m_data, src, get_size_in_bytes(), cudaMemcpyDefault);
}

void NDArray::write_data(const gbkfit::NDArray* src)
{
    if(const gbkfit::cuda::NDArray* src_array = dynamic_cast<const gbkfit::cuda::NDArray*>(src))
    {
        write_data(src_array);
    }
    else
    {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void NDArray::write_data(const gbkfit::cuda::NDArray* src)
{
    write_data(src->get_cuda_ptr());
}

ndarray_device::ndarray_device(const NDShape& shape)
    : gbkfit::cuda::NDArray(shape)
{
    cudaMalloc((void**)&m_data, get_size_in_bytes());
}

ndarray_device::~ndarray_device()
{
    cudaFree(m_data);
}

ndarray_pinned::ndarray_pinned(const NDShape& shape)
    : gbkfit::cuda::NDArray(shape)
{
    cudaHostAlloc((void**)&m_data, get_size_in_bytes(), cudaHostAllocDefault);
}

ndarray_pinned::~ndarray_pinned()
{
    cudaFreeHost(m_data);
}

ndarray_managed::ndarray_managed(const NDShape& shape)
    : gbkfit::cuda::NDArray(shape)
{
    cudaMallocManaged((void**)&m_data, get_size_in_bytes(), cudaMemAttachGlobal);
}

ndarray_managed::~ndarray_managed()
{
    cudaFree(m_data);
}

} // namespace cuda
} // namespace gbkfit
