
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/memory_buffer_malloc.hpp"

namespace gbkfit {

ndarray_host::ndarray_host(const ndshape& shape)
    : ndarray(shape)
{
    m_memory_buffer = new memory_buffer_malloc(shape.get_dim_length_product()*sizeof(value_type));
}

ndarray_host::ndarray_host(const ndshape& shape, const_pointer data)
    : ndarray_host(shape)
{
    m_memory_buffer->write_data(data);
}

ndarray_host::~ndarray_host()
{
    delete m_memory_buffer;
}

ndarray_host::pointer ndarray_host::get_host_ptr(void)
{
    memory_buffer_host* memory_buffer = reinterpret_cast<memory_buffer_host*>(m_memory_buffer);
    return reinterpret_cast<pointer>(memory_buffer->get_host_ptr());
}

ndarray_host::const_pointer ndarray_host::get_host_ptr(void) const
{
    const memory_buffer_host* memory_buffer =reinterpret_cast<const memory_buffer_host*>(m_memory_buffer);
    return reinterpret_cast<const_pointer>(memory_buffer->get_host_ptr());
}

void ndarray_host::read_data(pointer dst) const
{
    m_memory_buffer->read_data(dst);
}

void ndarray_host::write_data(const_pointer data)
{
    m_memory_buffer->write_data(data);
}

void ndarray_host::copy_data(const ndarray* src)
{
    if(const ndarray_host* src_array = reinterpret_cast<const ndarray_host*>(src))
    {
        copy_data(src_array);
    }
    else
    {
        throw std::runtime_error("trying to copy data from incompatible devices");
    }
}

void ndarray_host::copy_data(const ndarray_host* src)
{
    src->read_data(get_host_ptr());
}


} // namespace gbkfit
