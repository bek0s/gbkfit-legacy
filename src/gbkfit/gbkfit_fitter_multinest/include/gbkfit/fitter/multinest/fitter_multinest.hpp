#pragma once
#ifndef GBKFIT_FITTER_MULTINEST_FITTER_MULTINEST_HPP
#define GBKFIT_FITTER_MULTINEST_FITTER_MULTINEST_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitter {
namespace multinest {

class FitterMultinest : public Fitter
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
    static const int DEFAULT_SEED;
    static const int DEFAULT_OUTFILE;

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
    int m_seed;
    int m_outfile;

public:

    FitterMultinest(void);

    ~FitterMultinest();

    const std::string& get_type(void) const final;

    FitterResult* fit(const DModel* dmodel, const Params* params, const std::vector<Dataset*>& data) const final;

};

} // namespace multinest
} // namespace fitter
} // namespace gbkfit

#endif // GBKFIT_FITTER_MULTINEST_FITTER_MULTINEST_HPP
