
#include "gbkfit/nddataset.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {


nddataset::nddataset(void)
{
}

nddataset::~nddataset()
{
}

bool nddataset::has_data(const std::string& name) const
{
    return m_data_map.find(name) != m_data_map.end();
}

ndarray* nddataset::get_data(const std::string& name)
{
    auto iter = m_data_map.find(name);

    if(iter == m_data_map.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return iter->second;
}

const ndarray* nddataset::get_data(const std::string& name) const
{
    auto iter = m_data_map.find(name);

    if(iter == m_data_map.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return iter->second;
}

void nddataset::add_data(const std::string& name, ndarray* data)
{
    auto iter = m_data_map.find(name);

    if(iter != m_data_map.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    m_data_map.emplace(name,data);
}

void nddataset::__destroy(void)
{
    for(auto& data : m_data_map)
    {
        delete data.second;
    }
}


} // namespace gbkfit
