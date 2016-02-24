
#include "gbkfit/cuda/ndarray.hpp"
#include "gbkfit/ndarray_host.hpp"

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

NDArray::pointer NDArray::get_cuda_ptr(void)
{
    return m_data;
}

NDArray::const_pointer NDArray::get_cuda_ptr(void) const
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
    if (const gbkfit::cuda::NDArray* src_cuda = dynamic_cast<const gbkfit::cuda::NDArray*>(src))
    {
        write_data(src_cuda->get_cuda_ptr());
    }
    else if (const gbkfit::NDArrayHost* src_host = dynamic_cast<const gbkfit::NDArrayHost*>(src))
    {
        write_data(src_host->get_host_ptr());
    }
    else
    {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

NDArrayDevice::NDArrayDevice(const NDShape& shape)
    : gbkfit::cuda::NDArray(shape)
    , m_data_mapped(nullptr)
{
    cudaMalloc((void**)&m_data, get_size_in_bytes());
}

NDArrayDevice::~NDArrayDevice()
{
    cudaFree(m_data);
}

NDArrayDevice::pointer NDArrayDevice::map(void)
{
    m_data_mapped = new value_type[get_size()];
    read_data(m_data_mapped);
    return m_data_mapped;
}

void NDArrayDevice::unmap(void)
{
    write_data(m_data_mapped);
    delete [] m_data_mapped;
    m_data_mapped = nullptr;
}

NDArrayPinned::NDArrayPinned(const NDShape& shape)
    : gbkfit::cuda::NDArray(shape)
{
    cudaHostAlloc((void**)&m_data, get_size_in_bytes(), cudaHostAllocDefault);
}

NDArrayPinned::~NDArrayPinned()
{
    cudaFreeHost(m_data);
}

NDArrayPinned::pointer NDArrayPinned::map(void)
{
    return m_data;
}

void NDArrayPinned::unmap(void)
{
}

NDArrayManaged::NDArrayManaged(const NDShape& shape)
    : gbkfit::cuda::NDArray(shape)
{
    cudaMallocManaged((void**)&m_data, get_size_in_bytes(), cudaMemAttachGlobal);
}

NDArrayManaged::~NDArrayManaged()
{
    cudaFree(m_data);
}

NDArrayManaged::pointer NDArrayManaged::map(void)
{
    return m_data;
}

void NDArrayManaged::unmap(void)
{
}

} // namespace cuda
} // namespace gbkfit
