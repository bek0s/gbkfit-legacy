#pragma once
#ifndef GBKFIT_FITTERS_MPFIT_FITTER_FACTORY_MPFIT_HPP
#define GBKFIT_FITTERS_MPFIT_FITTER_FACTORY_MPFIT_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitters {
namespace mpfit {

class FitterFactoryMpfit : public FitterFactory
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    FitterFactoryMpfit(void);

    ~FitterFactoryMpfit(void);

    const std::string& get_type_name(void) const final;

    Fitter* create_fitter(const std::string& info) const final;

};

} // namespace mpfit
} // namespace fitters
} // namespace gbkfit

#endif // GBKFIT_FITTERS_MPFIT_FITTER_FACTORY_MPFIT_HPP
