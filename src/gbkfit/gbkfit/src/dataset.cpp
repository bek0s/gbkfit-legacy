
#include "gbkfit/dataset.hpp"
#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {

Dataset::Dataset(const std::string& name)
    : m_name(name)
    , m_data(nullptr)
{
}

const std::string& Dataset::get_name(void) const
{
    return m_name;
}

Dataset::Dataset(const std::string& name, NDArray* data, NDArray* errors, NDArray* mask)
    : m_name(name)
    , m_data(data)
    , m_errors(errors)
    , m_mask(mask)
{
    if (!m_errors)
        m_errors = new NDArrayHost(data->get_shape(), 1);
    if (!m_mask)
        m_mask = new NDArrayHost(data->get_shape(), 1);
}

Dataset::~Dataset()
{
    delete m_data;
    delete m_errors;
    delete m_mask;
}

NDArray* Dataset::get_data(void) const
{
    return m_data;
}

NDArray* Dataset::get_errors(void) const
{
    return m_errors;
}

NDArray* Dataset::get_mask(void) const
{
    return m_mask;
}

} // namespace gbkfit
