#pragma once
#ifndef GBKFIT_FITTER_MPFIT_FITTER_MPFIT_FACTORY_HPP
#define GBKFIT_FITTER_MPFIT_FITTER_MPFIT_FACTORY_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitter {
namespace mpfit {

class FitterMpfitFactory : public FitterFactory
{

public:

    static const std::string FACTORY_TYPE;

public:

    FitterMpfitFactory(void);

    ~FitterMpfitFactory(void);

    const std::string& get_type(void) const override final;

    Fitter* create(const std::string& info) const override final;

    void destroy(Fitter* fitter) const override final;

};

} // namespace mpfit
} // namespace fitter
} // namespace gbkfit

#endif // GBKFIT_FITTER_MPFIT_FITTER_MPFIT_FACTORY_HPP
