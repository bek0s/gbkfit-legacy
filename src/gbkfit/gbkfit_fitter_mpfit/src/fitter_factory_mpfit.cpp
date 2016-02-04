
#include "gbkfit/fitters/mpfit/fitter_factory_mpfit.hpp"
#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace fitters {
namespace mpfit {

const std::string FitterFactoryMpfit::FACTORY_TYPE_NAME = "gbkfit.fitter.mpfit";

FitterFactoryMpfit::FitterFactoryMpfit(void)
{
}

FitterFactoryMpfit::~FitterFactoryMpfit()
{
}

const std::string& FitterFactoryMpfit::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

Fitter* FitterFactoryMpfit::create_fitter(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    FitterMpfit* fitter = new FitterMpfit();
    fitter->m_ftol          = info_ptree.get<double>("ftol");
    fitter->m_xtol          = info_ptree.get<double>("xtol");
    fitter->m_gtol          = info_ptree.get<double>("gtol");
    fitter->m_epsfcn        = info_ptree.get<double>("epsfcn");
    fitter->m_stepfactor    = info_ptree.get<double>("stepfactor");
    fitter->m_covtol        = info_ptree.get<double>("covtol");
    fitter->m_maxiter       = info_ptree.get<int>("maxiter");
    fitter->m_maxfev        = info_ptree.get<int>("maxfev");
    fitter->m_nprint        = info_ptree.get<int>("nprint");
    fitter->m_douserscale   = info_ptree.get<int>("douserscale");
    fitter->m_nofinitecheck = info_ptree.get<int>("nofinitecheck");

    return fitter;
}

} // namespace mpfit
} // namespace fitters
} // namespace gbkfit
