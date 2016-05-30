
#include "gbkfit/dmodel/scube/scube_cuda_factory.hpp"
#include "gbkfit/dmodel/scube/scube_cuda.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

const std::string SCubeCudaFactory::FACTORY_TYPE = "gbkfit.dmodel.scube_cuda";

const std::string& SCubeCudaFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

DModel* SCubeCudaFactory::create(const std::string& info,
                                 const std::vector<int>& shape,
                                 const Instrument* instrument) const
{
    return nullptr;
}

void SCubeCudaFactory::destroy(DModel* dmodel) const
{
    if (dmodel->get_type() != get_type())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    delete dmodel;
}

} // namespace scube
} // namespace dmodel
} // namespace gbkfit
