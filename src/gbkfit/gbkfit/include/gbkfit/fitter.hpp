#pragma once
#ifndef GBKFIT_FITTER_HPP
#define GBKFIT_FITTER_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit
{

//!
//! \brief The fitter class
//!
class Fitter
{

public:

    Fitter(void);

    virtual ~Fitter();

    virtual const std::string& get_type_name(void) const = 0;

    virtual void fit(Model* model, const std::map<std::string,Dataset*>& data, Parameters& params_info) const = 0;

}; // class fitter

//!
//! \brief The fitter_factory class
//!
class FitterFactory
{

public:

    FitterFactory(void);

    virtual ~FitterFactory();

    virtual const std::string& get_type_name(void) const = 0;

    virtual Fitter* create_fitter(const std::string& info) const = 0;

}; //  class fitter_factory

} // namespace gbkfit

#endif // GBKFIT_FITTER_HPP
