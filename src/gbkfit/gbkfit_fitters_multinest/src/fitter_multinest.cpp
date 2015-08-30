
#include "gbkfit/fitters/multinest/fitter_multinest.hpp"
#include <multinest.h>

namespace gbkfit {
namespace fitters {
namespace multinest {


fitter_multinest::fitter_multinest(void)
{
}

fitter_multinest::~fitter_multinest()
{
}

const std::string& fitter_multinest::get_type_name(void) const
{
    return fitter_factory_multinest::FACTORY_TYPE_NAME;
}

void fitter_multinest::fit(model* model, const std::map<std::string,nddataset*>& data, parameters_fit_info& params_info) const
{
}


const std::string fitter_factory_multinest::FACTORY_TYPE_NAME = "gbkfit.fitters.multinest";

fitter_factory_multinest::fitter_factory_multinest(void)
    : fitter_factory()
{
}

fitter_factory_multinest::~fitter_factory_multinest()
{
}

const std::string& fitter_factory_multinest::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

fitter* fitter_factory_multinest::create_fitter(const std::string& info) const
{
    return new fitter_multinest();
}


} // namespace multinest
} // namespace fitters
} // namespace gbkfit
