
#include "gbkfit/ndarray.hpp"

namespace gbkfit {


ndarray::ndarray(const ndshape& shape)
    : m_shape(shape)
{
}

ndarray::~ndarray()
{
}

const ndshape& ndarray::get_shape(void) const
{
    return m_shape;
}


} // namespace gbkfit
