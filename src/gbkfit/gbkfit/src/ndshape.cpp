
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

NDShape::size_type NDShape::get_dim_count(void) const
{
    return m_shape.size();
}

NDShape::value_type NDShape::get_dim_length(size_type idx) const
{
    return m_shape[idx];
}

NDShape::value_type NDShape::get_dim_length_product(void) const
{
    return m_shape.size() ? std::accumulate(m_shape.begin(),m_shape.end(),static_cast<size_type>(1),std::multiplies<size_type>()) : 0;
}

const std::vector<NDShape::value_type>& NDShape::as_vector(void) const
{
    return m_shape;
}

const NDShape::value_type& NDShape::operator[](size_type idx) const
{
    return m_shape[idx];
}

NDShape::value_type& NDShape::operator[](size_type idx)
{
    return m_shape[idx];
}

bool NDShape::operator==(const NDShape& rhs) const
{
    return m_shape == rhs.m_shape;
}

bool NDShape::operator!=(const NDShape& rhs) const
{
    return m_shape != rhs.m_shape;
}


} // namespace gbkfit
