
#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {

NDArrayHost::NDArrayHost(const NDShape& shape)
    : NDArray(shape)
{
    m_data = new value_type[get_size()];
}

NDArrayHost::NDArrayHost(const NDShape &shape, const value_type& value)
    : NDArrayHost(shape)
{
    std::fill_n(m_data, get_size(), value);
}

NDArrayHost::~NDArrayHost()
{
    delete [] m_data;
}

NDArrayHost::pointer NDArrayHost::get_host_ptr(void)
{
    return m_data;
}

NDArrayHost::const_pointer NDArrayHost::get_host_ptr(void) const
{
    return m_data;
}

void NDArrayHost::read_data(pointer dst) const
{
    std::copy_n(m_data, get_size(), dst);
}

void NDArrayHost::write_data(const_pointer src)
{
    std::copy_n(src, get_size(), m_data);
}

void NDArrayHost::write_data(const NDArray* src)
{
    src->read_data(get_host_ptr());
}

NDArrayHost::pointer NDArrayHost::map(void)
{
    return m_data;
}

void NDArrayHost::unmap(void)
{
}

} // namespace gbkfit
