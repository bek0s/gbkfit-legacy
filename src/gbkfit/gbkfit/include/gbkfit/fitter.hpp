#pragma once
#ifndef GBKFIT_FITTER_HPP
#define GBKFIT_FITTER_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit
{

class Fitter
{

public:

    Fitter(void);

    virtual ~Fitter();

    virtual const std::string& get_type(void) const = 0;

    virtual FitterResult* fit(const DModel* dmodel,
                              const Parameters* params,
                              const std::vector<Dataset*>& data) const = 0;

};

class FitterFactory
{

public:

    FitterFactory(void);

    virtual ~FitterFactory();

    virtual const std::string& get_type(void) const = 0;

    virtual Fitter* create(const std::string& info) const = 0;

    virtual void destroy(Fitter* fitter) const = 0;

};

} // namespace gbkfit

#endif // GBKFIT_FITTER_HPP
