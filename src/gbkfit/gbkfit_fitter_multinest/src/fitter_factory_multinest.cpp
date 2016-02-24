
#include "gbkfit/fitters/multinest/fitter_factory_multinest.hpp"
#include "gbkfit/fitters/multinest/fitter_multinest.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace fitters {
namespace multinest {

const std::string FitterFactoryMultinest::FACTORY_TYPE_NAME = "gbkfit.fitter.multinest";

FitterFactoryMultinest::FitterFactoryMultinest(void)
{
}

FitterFactoryMultinest::~FitterFactoryMultinest()
{
}

const std::string& FitterFactoryMultinest::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

Fitter* FitterFactoryMultinest::create_fitter(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    FitterMultinest* fitter = new FitterMultinest();
    if (info_ptree.count("efr") > 0)
        fitter->m_efr     = info_ptree.get<double>("efr");
    if (info_ptree.count("tol") > 0)
        fitter->m_tol     = info_ptree.get<double>("tol");
    if (info_ptree.count("ztol") > 0)
        fitter->m_ztol    = info_ptree.get<double>("ztol");
    if (info_ptree.count("logzero") > 0)
        fitter->m_logzero = info_ptree.get<double>("logzero");
    if (info_ptree.count("is") > 0)
        fitter->m_is      = info_ptree.get<int>("is");
    if (info_ptree.count("mmodal") > 0)
        fitter->m_mmodal  = info_ptree.get<int>("mmodal");
    if (info_ptree.count("ceff") > 0)
        fitter->m_ceff    = info_ptree.get<int>("ceff");
    if (info_ptree.count("nlive") > 0)
        fitter->m_nlive   = info_ptree.get<int>("nlive");
    if (info_ptree.count("maxiter") > 0)
        fitter->m_maxiter = info_ptree.get<int>("maxiter");

    return fitter;
}

} // namespace multinest
} // namespace fitters
} // namespace gbkfit
