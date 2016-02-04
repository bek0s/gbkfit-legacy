#pragma once
#ifndef GBKFIT_FITTERS_MULTINEST_FITTER_MULTINEST_HPP
#define GBKFIT_FITTERS_MULTINEST_FITTER_MULTINEST_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitters {
namespace multinest {

class FitterMultinest : public gbkfit::Fitter
{

public:

    static const double DEFAULT_EFR;
    static const double DEFAULT_TOL;
    static const double DEFAULT_ZTOL;
    static const double DEFAULT_LOGZERO;
    static const int DEFAULT_IS;
    static const int DEFAULT_MMODAL;
    static const int DEFAULT_CEFF;
    static const int DEFAULT_NLIVE;
    static const int DEFAULT_MAXITER;

public:

    double m_efr;
    double m_tol;
    double m_ztol;
    double m_logzero;
    int m_is;
    int m_mmodal;
    int m_ceff;
    int m_nlive;
    int m_maxiter;

public:

    FitterMultinest(void);

    ~FitterMultinest();

    const std::string& get_type_name(void) const final;

    void fit(Model* Model, const std::map<std::string,Dataset*>& data, Parameters& params_info) const final;

};

} // namespace multinest
} // namespace fitters
} // namespace gbkfit

#endif // GBKFIT_FITTERS_MULTINEST_FITTER_MULTINEST_HPP
