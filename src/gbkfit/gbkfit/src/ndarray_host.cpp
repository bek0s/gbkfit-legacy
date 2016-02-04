
#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {

NDArrayHost::NDArrayHost(const NDShape& shape)
    : NDArray(shape)
{
    m_data = new value_type[get_size()];
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
    if(const NDArrayHost* src_array = dynamic_cast<const NDArrayHost*>(src))
    {
        write_data(src_array);
    }
    else
    {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void NDArrayHost::write_data(const NDArrayHost* src)
{
    src->read_data(get_host_ptr());
}

} // namespace gbkfit
