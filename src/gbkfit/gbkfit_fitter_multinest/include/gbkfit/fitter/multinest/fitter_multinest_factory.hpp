#pragma once
#ifndef GBKFIT_FITTER_MULTINEST_FITTER_MULTINEST_FACTORY_HPP
#define GBKFIT_FITTER_MULTINEST_FITTER_MULTINEST_FACTORY_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitter {
namespace multinest {

class FitterMultinestFactory : public FitterFactory
{

public:

    static const std::string FACTORY_TYPE;

public:

    FitterMultinestFactory(void);

    ~FitterMultinestFactory();

    const std::string& get_type(void) const override final;

    Fitter* create(const std::string& info) const override final;

    void destroy(Fitter* fitter) const override final;

};

} // namespace multinest
} // namespace fitter
} // namespace gbkfit

#endif // GBKFIT_FITTER_MULTINEST_FITTER_MULTINEST_FACTORY_HPP
