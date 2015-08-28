#pragma once
#ifndef GBKFIT_FITTERS_MPFIT_FITTER_MPFIT_HPP
#define GBKFIT_FITTERS_MPFIT_FITTER_MPFIT_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitters {
namespace mpfit {


//!
//! \brief The fitter_mpfit class
//!
class fitter_mpfit : public fitter
{

public:

    fitter_mpfit(void);

    ~fitter_mpfit();

    virtual const std::string& get_type_name(void) const final;

    void fit(model* model, const std::map<std::string,nddataset*>& data, model_parameters_fit_info& params_info) const final;

}; // class fitter_mpfit

//!
//! \brief The fitter_factory_mpfit class
//!
class fitter_factory_mpfit : public fitter_factory
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    fitter_factory_mpfit(void);

    ~fitter_factory_mpfit(void);

    const std::string& get_type_name(void) const final;

    fitter* create_fitter(const std::string& info) const final;

}; // class fitter_factory_mpfit


} // namespace mpfit
} // namespace fitters
} // namespace gbkfit

#endif  //  GBKFIT_FITTERS_MPFIT_FITTER_MPFIT_HPP
