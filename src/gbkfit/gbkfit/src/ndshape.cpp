
#include "gbkfit/ndshape.hpp"

namespace gbkfit {


ndshape::ndshape(const std::vector<value_type>& shape)
    : m_shape(shape)
{
}

ndshape::ndshape(const std::initializer_list<value_type>& shape)
    : ndshape(std::vector<value_type>(shape))
{
}

ndshape::~ndshape()
{
}

ndshape::size_type ndshape::get_dim_count(void) const
{
    return m_shape.size();
}

ndshape::value_type ndshape::get_dim_length(size_type idx) const
{
    return m_shape[idx];
}

ndshape::value_type ndshape::get_dim_length_product(void) const
{
    return std::accumulate(m_shape.begin(),m_shape.end(),static_cast<size_type>(1),std::multiplies<size_type>());
}

const std::vector<ndshape::value_type>& ndshape::get_as_vector(void) const
{
    return m_shape;
}

const ndshape::value_type& ndshape::operator[](size_type idx) const
{
    return m_shape[idx];
}

ndshape::value_type& ndshape::operator[](size_type idx)
{
    return m_shape[idx];
}

bool ndshape::operator==(const ndshape& rhs) const
{
    return m_shape == rhs.m_shape;
}

bool ndshape::operator!=(const ndshape& rhs) const
{
    return m_shape != rhs.m_shape;
}


} // namespace gbkfit
