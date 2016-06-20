#pragma once
#ifndef GBKFIT_FITTER_MPFIT_FITTER_MPFIT_HPP
#define GBKFIT_FITTER_MPFIT_FITTER_MPFIT_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitter {
namespace mpfit {

class FitterMpfit : public Fitter
{

public:

    static const double DEFAULT_FTOL;
    static const double DEFAULT_XTOL;
    static const double DEFAULT_GTOL;
    static const double DEFAULT_EPSFCN;
    static const double DEFAULT_STEPFACTOR;
    static const double DEFAULT_COVTOL;
    static const int DEFAULT_MAXITER;
    static const int DEFAULT_MAXFEV;
    static const int DEFAULT_NPRINT;
    static const int DEFAULT_DOUSERSCALE;
    static const int DEFAULT_NOFINITECHECK;

public:

    double m_ftol;
    double m_xtol;
    double m_gtol;
    double m_epsfcn;
    double m_stepfactor;
    double m_covtol;
    int m_maxiter;
    int m_maxfev;
    int m_nprint;
    int m_douserscale;
    int m_nofinitecheck;

public:

    FitterMpfit(void);

    ~FitterMpfit();

    const std::string& get_type(void) const override final;

    FitterResult* fit(const DModel* dmodel,
                      const Params* params,
                      const std::vector<Dataset*>& data) const final;

};

} // namespace mpfit
} // namespace fitter
} // namespace gbkfit

#endif // GBKFIT_FITTER_MPFIT_FITTER_MPFIT_HPP
