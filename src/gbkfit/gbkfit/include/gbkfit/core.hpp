#pragma once
#ifndef GBKFIT_CORE_HPP
#define GBKFIT_CORE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


//!
//! \brief The core class
//!
class core
{

private:

    std::map<std::string,model_factory*> m_model_factories;
    std::map<std::string,fitter_factory*> m_fitter_factories;

public:

    //!
    //! \brief add_model_factory
    //! \param factory
    //!
    void add_model_factory(model_factory* factory);

    //!
    //! \brief add_fitter_factory
    //! \param factory
    //!
    void add_fitter_factory(fitter_factory* factory);

    //!
    //! \brief create_model
    //! \param info
    //! \return
    //!
    model* create_model(const std::string& info) const;

    //!
    //! \brief create_fitter
    //! \param info
    //! \return
    //!
    fitter* create_fitter(const std::string& info) const;

    //!
    //! \brief create_parameters_fit_info
    //! \param info
    //! \return
    //!
    parameters_fit_info* create_parameters_fit_info(const std::string& info) const;

    //!
    //! \brief create_dataset
    //! \param info
    //! \return
    //!
    nddataset* create_dataset(const std::string& info) const;

    //!
    //! \brief create_datasets
    //! \param info
    //! \return
    //!
    std::map<std::string,nddataset*> create_datasets(const std::string& info) const;


}; // class core


} // namespace gbkfit

#endif // GBKFIT_CORE_HPP
