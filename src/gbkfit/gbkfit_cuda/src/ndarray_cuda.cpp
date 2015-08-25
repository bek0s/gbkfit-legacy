
#include "gbkfit/cuda/ndarray_cuda.hpp"

namespace gbkfit {
namespace cuda {


ndarray_cuda::ndarray_cuda(const ndshape& shape)
    : ndarray(shape)
{
}

ndarray_cuda::~ndarray_cuda()
{
}

ndarray_cuda::pointer ndarray_cuda::get_cuda_ptr(void)
{
    memory_buffer_cuda* memory_buffer = reinterpret_cast<memory_buffer_cuda*>(m_memory_buffer);
    return reinterpret_cast<pointer>(memory_buffer->get_cuda_ptr());
}

ndarray_cuda::const_pointer ndarray_cuda::get_cuda_ptr(void) const
{
    const memory_buffer_cuda* memory_buffer =reinterpret_cast<const memory_buffer_cuda*>(m_memory_buffer);
    return reinterpret_cast<const_pointer>(memory_buffer->get_cuda_ptr());
}

void ndarray_cuda::read_data(pointer dst) const
{
    m_memory_buffer->read_data(dst);
}

void ndarray_cuda::write_data(const_pointer data)
{
    m_memory_buffer->write_data(data);
}

void ndarray_cuda::copy_data(const ndarray* src)
{
    if(const ndarray_cuda* src_array = reinterpret_cast<const ndarray_cuda*>(src))
    {
        copy_data(src_array);
    }
    else
    {
        // throw
    }
}

void ndarray_cuda::copy_data(const ndarray_cuda* src)
{
    src->read_data(get_cuda_ptr());
}

ndarray_cuda_device::ndarray_cuda_device(const ndshape& shape)
    : ndarray_cuda(shape)
{
    m_memory_buffer = new memory_buffer_cuda_device(shape.get_dim_length_product()*sizeof(value_type));
}

ndarray_cuda_device::ndarray_cuda_device(const ndshape& shape, const_pointer data)
    : ndarray_cuda_device(shape)
{
    m_memory_buffer->write_data(data);
}

ndarray_cuda_device::~ndarray_cuda_device()
{
    delete m_memory_buffer;
}

ndarray_cuda_pinned::ndarray_cuda_pinned(const ndshape& shape)
    : ndarray_cuda(shape)
{
    m_memory_buffer = new memory_buffer_cuda_pinned(shape.get_dim_length_product()*sizeof(value_type));
}

ndarray_cuda_pinned::ndarray_cuda_pinned(const ndshape& shape, const_pointer data)
    : ndarray_cuda_pinned(shape)
{
    m_memory_buffer->write_data(data);
}

ndarray_cuda_pinned::~ndarray_cuda_pinned()
{
    delete m_memory_buffer;
}

ndarray_cuda_pinned_wc::ndarray_cuda_pinned_wc(const ndshape& shape)
    : ndarray_cuda(shape)
{
    m_memory_buffer = new memory_buffer_cuda_pinned_wc(shape.get_dim_length_product()*sizeof(value_type));
}

ndarray_cuda_pinned_wc::ndarray_cuda_pinned_wc(const ndshape& shape, const_pointer data)
    : ndarray_cuda_pinned_wc(shape)
{
    m_memory_buffer->write_data(data);
}

ndarray_cuda_pinned_wc::~ndarray_cuda_pinned_wc()
{
    delete m_memory_buffer;
}


} // namespace cuda
} // namespace gbkfit
