#pragma once
#ifndef GBKFIT_FITTERS_MULTINEST_FITTER_FACTORY_MULTINEST_HPP
#define GBKFIT_FITTERS_MULTINEST_FITTER_FACTORY_MULTINEST_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitters {
namespace multinest {

class FitterFactoryMultinest : public gbkfit::FitterFactory
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    FitterFactoryMultinest(void);

    ~FitterFactoryMultinest();

    const std::string& get_type_name(void) const final;

    Fitter* create_fitter(const std::string& info) const final;

};

} // namespace multinest
} // namespace fitters
} // namespace gbkfit

#endif // GBKFIT_FITTERS_MULTINEST_FITTER_FACTORY_MULTINEST_HPP
