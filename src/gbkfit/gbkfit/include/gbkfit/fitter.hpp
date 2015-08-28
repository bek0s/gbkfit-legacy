#pragma once
#ifndef GBKFIT_FITTER_HPP
#define GBKFIT_FITTER_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit
{


//!
//! \brief The fitter class
//!
class fitter
{

public:

    fitter(void);

    virtual ~fitter();

    virtual const std::string& get_type_name(void) const = 0;

    virtual void fit(model* model, const std::vector<nddataset*>& data, model_parameters_fit_info& params_info) const = 0;


}; // class fitter


//!
//! \brief The fitter_factory class
//!
class fitter_factory
{

public:

    fitter_factory(void);

    virtual ~fitter_factory();

    virtual const std::string& get_type_name(void) const = 0;

    virtual fitter* create_fitter(const std::string& info) const = 0;

}; //  class fitter_factory


} // namespace gbkfit

#endif // GBKFIT_FITTER_HPP
