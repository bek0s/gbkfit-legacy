#pragma once
#ifndef GBKFIT_CORE_HPP
#define GBKFIT_CORE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

//!
//! \brief The Core class
//!
class Core
{

private:

    std::map<std::string,ModelFactory*> m_model_factories;
    std::map<std::string,FitterFactory*> m_fitter_factories;

public:


    void add_model_factory(ModelFactory* factory);

    void add_fitter_factory(FitterFactory* factory);

    Model* create_model(const std::string& info) const;

    Fitter* create_fitter(const std::string& info) const;

    Parameters* create_parameters(const std::string& info) const;

    Dataset* create_dataset(const std::string& info) const;

    std::map<std::string, Dataset*> create_datasets(const std::string& info) const;

    Instrument* create_instrument(const std::string& info) const;

}; // class Core

} // namespace gbkfit

#endif // GBKFIT_CORE_HPP
