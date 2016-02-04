
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
    fitter->m_efr     = info_ptree.get<double>("efr");
    fitter->m_tol     = info_ptree.get<double>("tol");
    fitter->m_ztol    = info_ptree.get<double>("ztol");
    fitter->m_logzero = info_ptree.get<double>("logzero");
    fitter->m_is      = info_ptree.get<int>("is");
    fitter->m_mmodal  = info_ptree.get<int>("mmodal");
    fitter->m_ceff    = info_ptree.get<int>("ceff");
    fitter->m_nlive   = info_ptree.get<int>("nlive");
    fitter->m_maxiter = info_ptree.get<int>("maxier");

    return fitter;
}

} // namespace multinest
} // namespace fitters
} // namespace gbkfit
