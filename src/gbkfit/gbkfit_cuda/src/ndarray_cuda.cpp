
#include "gbkfit/cuda/ndarray_cuda.hpp"

#include <cuda_runtime_api.h>

namespace gbkfit {
namespace cuda {

ndarray_cuda::ndarray_cuda(const ndshape& shape)
    : ndarray(shape)
    , m_data(nullptr)
{
}

ndarray_cuda::~ndarray_cuda()
{
}

ndarray_cuda::pointer ndarray_cuda::get_cuda_ptr(void)
{
    return m_data;
}

ndarray_cuda::const_pointer ndarray_cuda::get_cuda_ptr(void) const
{
    return m_data;
}

void ndarray_cuda::read_data(pointer dst) const
{
    const_pointer src = m_data;
    std::size_t size = get_shape().get_dim_length_product()*sizeof(value_type);
    cudaMemcpy(dst,src,size,cudaMemcpyDefault);
}

void ndarray_cuda::write_data(const_pointer src)
{
    pointer dst = m_data;
    std::size_t size = get_shape().get_dim_length_product()*sizeof(value_type);
    cudaMemcpy(dst,src,size,cudaMemcpyDefault);
}

void ndarray_cuda::copy_data(const ndarray* src)
{
    if(const ndarray_cuda* src_array = reinterpret_cast<const ndarray_cuda*>(src))
    {
        copy_data(src_array);
    }
    else
    {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void ndarray_cuda::copy_data(const ndarray_cuda* src)
{
    write_data(src->get_cuda_ptr());
}

ndarray_cuda_device::ndarray_cuda_device(const ndshape& shape)
    : ndarray_cuda(shape)
{
    cudaMalloc((void**)&m_data,shape.get_dim_length_product());
}

ndarray_cuda_device::ndarray_cuda_device(const ndshape& shape, const_pointer data)
    : ndarray_cuda_device(shape)
{
    ndarray_cuda::write_data(data);
}

ndarray_cuda_device::~ndarray_cuda_device()
{
    cudaFree(m_data);
}

ndarray_cuda_pinned::ndarray_cuda_pinned(const ndshape& shape)
    : ndarray_cuda(shape)
{
    unsigned int flags = cudaHostAllocDefault;
    cudaHostAlloc((void**)&m_data,shape.get_dim_length_product(),flags);
}

ndarray_cuda_pinned::ndarray_cuda_pinned(const ndshape& shape, const_pointer data)
    : ndarray_cuda_pinned(shape)
{
    ndarray_cuda::write_data(data);
}

ndarray_cuda_pinned::~ndarray_cuda_pinned()
{
    cudaFreeHost(m_data);
}

ndarray_cuda_pinned_wc::ndarray_cuda_pinned_wc(const ndshape& shape)
    : ndarray_cuda(shape)
{
    unsigned int flags = cudaHostAllocWriteCombined;
    cudaHostAlloc((void**)&m_data,shape.get_dim_length_product(),flags);
}

ndarray_cuda_pinned_wc::ndarray_cuda_pinned_wc(const ndshape& shape, const_pointer data)
    : ndarray_cuda_pinned_wc(shape)
{
    ndarray_cuda::write_data(data);
}

ndarray_cuda_pinned_wc::~ndarray_cuda_pinned_wc()
{
    cudaFreeHost(m_data);
}

} // namespace cuda
} // namespace gbkfit
