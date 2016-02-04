
#include "gbkfit/ndarray.hpp"

namespace gbkfit {

NDArray::NDArray(const NDShape& shape)
    : m_shape(shape)
{
}

NDArray::~NDArray()
{
}

const NDShape& NDArray::get_shape(void) const
{
    return m_shape;
}

NDArray::size_type NDArray::get_size(void) const
{
    return m_shape.get_dim_length_product();
}

NDArray::size_type NDArray::get_size_in_bytes(void) const
{
    return get_size() * sizeof(NDArray::value_type);
}

} // namespace gbkfit
