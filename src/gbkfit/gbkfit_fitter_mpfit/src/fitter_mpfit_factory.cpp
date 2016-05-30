
#include "gbkfit/fitter/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitter/mpfit/fitter_mpfit_factory.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <jsoncpp/json/json.h>

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace fitter {
namespace mpfit {

const std::string FitterMpfitFactory::FACTORY_TYPE = "gbkfit.fitter.mpfit";

FitterMpfitFactory::FitterMpfitFactory(void)
{
}

FitterMpfitFactory::~FitterMpfitFactory()
{
}

const std::string& FitterMpfitFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

Fitter* FitterMpfitFactory::create(const std::string& info) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    FitterMpfit* fitter = new FitterMpfit();

    if (info_root.count("ftol"))
        fitter->m_ftol          = info_root["ftol"].get<double>();
    if (info_root.count("xtol"))
        fitter->m_xtol          = info_root["xtol"].get<double>();
    if (info_root.count("gtol"))
        fitter->m_gtol          = info_root["gtol"].get<double>();
    if (info_root.count("epsfcn"))
        fitter->m_epsfcn        = info_root["epsfcn"].get<double>();
    if (info_root.count("stepfactor"))
        fitter->m_stepfactor    = info_root["stepfactor"].get<double>();
    if (info_root.count("covtol"))
        fitter->m_covtol        = info_root["covtol"].get<double>();
    if (info_root.count("maxiter"))
        fitter->m_maxiter       = info_root["maxiter"].get<int>();
    if (info_root.count("maxfev"))
        fitter->m_maxfev        = info_root["maxfev"].get<int>();
    if (info_root.count("nprint"))
        fitter->m_nprint        = info_root["nprint"].get<int>();
    if (info_root.count("douserscale"))
        fitter->m_douserscale   = info_root["douserscale"].get<int>();
    if (info_root.count("nofinitecheck"))
        fitter->m_nofinitecheck = info_root["nofinitecheck"].get<int>();

    return fitter;
}

void FitterMpfitFactory::destroy(Fitter* fitter) const
{
    delete fitter;
}

} // namespace mpfit
} // namespace fitters
} // namespace gbkfit
