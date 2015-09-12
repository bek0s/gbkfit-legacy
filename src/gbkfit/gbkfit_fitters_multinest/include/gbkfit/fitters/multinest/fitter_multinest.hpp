#pragma once
#ifndef GBKFIT_FITTERS_MULTINEST_FITTER_MULTINEST_HPP
#define GBKFIT_FITTERS_MULTINEST_FITTER_MULTINEST_HPP

#include "gbkfit/fitter.hpp"

namespace gbkfit {
namespace fitters {
namespace multinest {

//!
//! \brief The fitter_multinest class
//!
class fitter_multinest : public gbkfit::fitter
{

public:

    fitter_multinest(void);

    ~fitter_multinest();

    const std::string& get_type_name(void) const final;

    void fit(model* model, const std::map<std::string,nddataset*>& data, parameters_fit_info& params_info) const final;

}; // class fitter_multinest

//!
//! \brief The fitter_factory_multinest class
//!
class fitter_factory_multinest : public gbkfit::fitter_factory
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    fitter_factory_multinest(void);

    ~fitter_factory_multinest();

    const std::string& get_type_name(void) const final;

    fitter* create_fitter(const std::string& info) const final;

}; // class fitter_factory_multinest

} // namespace multinest
} // namespace fitters
} // namespace gbkfit

#endif // GBKFIT_FITTERS_MULTINEST_FITTER_MULTINEST_HPP
