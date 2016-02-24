
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
    if (info_ptree.count("ftol") > 0)
        fitter->m_ftol          = info_ptree.get<double>("ftol");
    if (info_ptree.count("xtol") > 0)
        fitter->m_xtol          = info_ptree.get<double>("xtol");
    if (info_ptree.count("gtol") > 0)
        fitter->m_gtol          = info_ptree.get<double>("gtol");
    if (info_ptree.count("epsfcn") > 0)
        fitter->m_epsfcn        = info_ptree.get<double>("epsfcn");
    if (info_ptree.count("stepfactor") > 0)
        fitter->m_stepfactor    = info_ptree.get<double>("stepfactor");
    if (info_ptree.count("covtol") > 0)
        fitter->m_covtol        = info_ptree.get<double>("covtol");
    if (info_ptree.count("maxiter") > 0)
        fitter->m_maxiter       = info_ptree.get<int>("maxiter");
    if (info_ptree.count("maxfev") > 0)
        fitter->m_maxfev        = info_ptree.get<int>("maxfev");
    if (info_ptree.count("nprint") > 0)
        fitter->m_nprint        = info_ptree.get<int>("nprint");
    if (info_ptree.count("douserscale") > 0)
        fitter->m_douserscale   = info_ptree.get<int>("douserscale");
    if (info_ptree.count("nofinitecheck") > 0)
        fitter->m_nofinitecheck = info_ptree.get<int>("nofinitecheck");

    return fitter;
}

} // namespace mpfit
} // namespace fitters
} // namespace gbkfit
