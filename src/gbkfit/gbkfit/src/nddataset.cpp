
#include "gbkfit/nddataset.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {

Dataset::Dataset(const std::string& type)
    : m_type(type)
{
}

Dataset::~Dataset()
{
}

const std::string& Dataset::get_type(void) const
{
    return m_type;
}

bool Dataset::has_data(const std::string& name) const
{
    return m_data_map.find(name) != m_data_map.end();
}

NDArray* Dataset::get_data(const std::string& name)
{
    auto iter = m_data_map.find(name);
    if (iter == m_data_map.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return iter->second;
}

const NDArray* Dataset::get_data(const std::string& name) const
{
    auto iter = m_data_map.find(name);
    if (iter == m_data_map.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return iter->second;
}

void Dataset::add_data(const std::string& name, NDArray* data)
{
    auto iter = m_data_map.find(name);
    if (iter != m_data_map.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    m_data_map.emplace(name,data);
}

} // namespace gbkfit
