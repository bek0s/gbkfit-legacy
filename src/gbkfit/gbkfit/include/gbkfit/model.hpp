#pragma once
#ifndef GBKFIT_MODEL_HPP
#define GBKFIT_MODEL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

//!
//! \brief The model class
//!
class Model
{

public:

    Model(void);

    virtual ~Model();

    virtual void initialize(int size_x, int size_y, int size_z, Instrument* Instrument) = 0;

    virtual const std::string& get_type_name(void) const = 0;

    virtual const std::vector<std::string>& get_parameter_names(void) const = 0;

    virtual const std::vector<float>& get_parameter_values(void) const = 0;

    virtual const std::map<std::string,NDArray*>& get_data(void) const = 0;

    virtual const std::map<std::string,NDArray*>& evaluate(const std::map<std::string,float>& parameters) = 0;

}; // class Model

//!
//! \brief The model_factory class
//!
class ModelFactory
{

public:

    ModelFactory(void);

    virtual ~ModelFactory();

    virtual const std::string& get_type_name(void) const = 0;

    virtual Model* create_model(const std::string& info) const = 0;

}; //  class ModelFactory

} // namespace gbkfit

#endif // GBKFIT_MODEL_HPP
