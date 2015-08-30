#pragma once
#ifndef GBKFIT_MODEL_HPP
#define GBKFIT_MODEL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


//!
//! \brief The model class
//!
class model
{

public:

    model(void);

    virtual ~model();

    virtual const std::string& get_type_name(void) const = 0;

    virtual const std::vector<std::string>& get_parameter_names(void) const = 0;

    virtual const std::vector<float>& get_parameter_values(void) const = 0;

    virtual const std::map<std::string,ndarray*>& get_data(void) const = 0;

    virtual const std::map<std::string,ndarray*>& evaluate(const std::map<std::string,float>& parameters) = 0;

}; // class model


//!
//! \brief The model_factory class
//!
class model_factory
{

public:

    model_factory(void);

    virtual ~model_factory();

    virtual const std::string& get_type_name(void) const = 0;

    virtual model* create_model(const std::string& info) const = 0;

}; //  class model_factory


} // namespace gbkfit

#endif // GBKFIT_MODEL_HPP
