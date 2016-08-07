
#include "gbkfit/fitter/multinest/fitter_multinest_factory.hpp"
#include "gbkfit/fitter/multinest/fitter_multinest.hpp"

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace fitter {
namespace multinest {

const std::string FitterMultinestFactory::FACTORY_TYPE = "gbkfit.fitter.multinest";

FitterMultinestFactory::FitterMultinestFactory(void)
{
}

FitterMultinestFactory::~FitterMultinestFactory()
{
}

const std::string& FitterMultinestFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

Fitter* FitterMultinestFactory::create(const std::string& info) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    FitterMultinest* fitter = new FitterMultinest();

    if (info_root.count("efr"))
        fitter->m_efr     = info_root["efr"].get<double>();
    if (info_root.count("tol"))
        fitter->m_tol     = info_root["tol"].get<double>();
    if (info_root.count("ztol"))
        fitter->m_ztol    = info_root["ztol"].get<double>();
    if (info_root.count("logzero"))
        fitter->m_logzero = info_root["logzero"].get<double>();
    if (info_root.count("is"))
        fitter->m_is    = info_root["is"].get<int>();
    if (info_root.count("mmodal"))
        fitter->m_mmodal  = info_root["mmodal"].get<int>();
    if (info_root.count("ceff"))
        fitter->m_maxiter = info_root["ceff"].get<int>();
    if (info_root.count("nlive"))
        fitter->m_nlive   = info_root["nlive"].get<int>();
    if (info_root.count("maxiter"))
        fitter->m_maxiter = info_root["maxiter"].get<int>();
    if (info_root.count("seed"))
        fitter->m_seed = info_root["seed"].get<int>();

    return fitter;
}

void FitterMultinestFactory::destroy(Fitter* fitter) const
{
    delete fitter;
}

} // namespace multinest
} // namespace fitters
} // namespace gbkfit
