
#include "gbkfit/nddataset.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {


nddataset::nddataset(void)
{
}

nddataset::~nddataset()
{
}

std::map<std::string,ndarray*>& nddataset::get(void)
{
    return m_data_map;
}

} // namespace gbkfit
