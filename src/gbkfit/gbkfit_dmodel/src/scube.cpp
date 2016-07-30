
#include "gbkfit/dmodel/scube/scube.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

SCube::SCube(void)
    : m_gmodel(nullptr)
{
}

const GModel* SCube::get_galaxy_model(void) const
{
    return m_gmodel;
}

void SCube::set_galaxy_model(const GModel* gmodel)
{
    m_gmodel = gmodel;
}

} // namespace scube
} // namesapce dmodel
} // namespace gbkfit
