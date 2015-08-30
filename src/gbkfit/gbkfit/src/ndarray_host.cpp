
#include "gbkfit/ndarray_host.hpp"

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
    return m_data;
}

ndarray_host::const_pointer ndarray_host::get_host_ptr(void) const
{
    return m_data;
}

void ndarray_host::read_data(pointer dst) const
{
    const_pointer src = m_data;
    std::size_t size = get_shape().get_dim_length_product();
    std::copy(src,src+size,dst);
}

void ndarray_host::write_data(const_pointer src)
{
    pointer dst = m_data;
    std::size_t size = get_shape().get_dim_length_product();
    std::copy(src,src+size,dst);
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
    std::size_t size = shape.get_dim_length_product()*sizeof(float);
    m_data = reinterpret_cast<pointer>(std::malloc(size));
}

ndarray_host_malloc::ndarray_host_malloc(const ndshape& shape, const_pointer data)
    : ndarray_host_malloc(shape)
{
    ndarray_host::write_data(data);
}

ndarray_host_malloc::ndarray_host_malloc(const ndarray& array)
    : ndarray_host_malloc(array.get_shape())
{
    ndarray_host::copy_data(&array);
}

ndarray_host_malloc::~ndarray_host_malloc()
{
    free(m_data);
}

ndarray_host_new::ndarray_host_new(const ndshape& shape)
    : ndarray_host(shape)
{
    std::size_t size = shape.get_dim_length_product();
    m_data = new value_type[size];
}

ndarray_host_new::ndarray_host_new(const ndshape& shape, const_pointer data)
    : ndarray_host_new(shape)
{
    ndarray_host::write_data(data);
}

ndarray_host_new::ndarray_host_new(const ndarray& array)
    : ndarray_host_new(array.get_shape())
{
    ndarray_host::copy_data(&array);
}

ndarray_host_new::~ndarray_host_new()
{
    delete [] m_data;
}

} // namespace gbkfit
