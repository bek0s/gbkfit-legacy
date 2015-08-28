
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/memory_buffer_malloc.hpp"

namespace gbkfit {


ndarray_host::ndarray_host(const ndshape& shape)
    : ndarray(shape)
{
}

ndarray_host::~ndarray_host()
{
}

ndarray_host::pointer ndarray_host::get_host_ptr(void)
{
    return reinterpret_cast<pointer>(m_data);
}

ndarray_host::const_pointer ndarray_host::get_host_ptr(void) const
{
    return reinterpret_cast<const_pointer>(m_data);
}

void ndarray_host::read_data(pointer dst) const
{
    const_pointer src = m_data;
    std::copy(src,src+get_shape().get_dim_length_product(),dst);
}

void ndarray_host::write_data(const_pointer src)
{
    pointer dst = m_data;
    std::copy(src,src+get_shape().get_dim_length_product(),dst);
}

void ndarray_host::copy_data(const ndarray* src)
{
    if(const ndarray_host* src_array = reinterpret_cast<const ndarray_host*>(src))
    {
        copy_data(src_array);
    }
    else
    {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void ndarray_host::copy_data(const ndarray_host* src)
{
    src->read_data(get_host_ptr());
}

ndarray_host_malloc::ndarray_host_malloc(const ndshape& shape)
    : ndarray_host(shape)
{
    m_data = reinterpret_cast<pointer>(std::malloc(shape.get_dim_length_product()*sizeof(float)));
}

ndarray_host_malloc::ndarray_host_malloc(const ndshape& shape, const_pointer data)
    : ndarray_host_malloc(shape)
{
    ndarray_host::write_data(data);
}

ndarray_host_malloc::~ndarray_host_malloc()
{
    free(m_data);
}

ndarray_host_new::ndarray_host_new(const ndshape& shape)
    : ndarray_host(shape)
{
    m_data = new value_type[shape.get_dim_length_product()];
}

ndarray_host_new::ndarray_host_new(const ndshape& shape, const_pointer data)
    : ndarray_host_new(shape)
{
    ndarray_host::write_data(data);
}

ndarray_host_new::~ndarray_host_new()
{
    delete [] m_data;
}


} // namespace gbkfit
