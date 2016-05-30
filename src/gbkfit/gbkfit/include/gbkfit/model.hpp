#pragma once
#ifndef GBKFIT_MODEL_HPP
#define GBKFIT_MODEL_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

class Model
{

public:

    Model(void);

    virtual ~Model();

    virtual void initialize(int size_x, int size_y, int size_z, Instrument* Instrument) = 0;

    virtual const std::string& get_type(void) const = 0;

    virtual const std::vector<std::string>& get_parameter_names(void) const = 0;

    virtual const std::map<std::string, NDArray*>& evaluate(const std::map<std::string,float>& parameters) const = 0;

};

class ModelFactory
{

public:

    ModelFactory(void);

    virtual ~ModelFactory();

    virtual const std::string& get_type(void) const = 0;

    virtual Model* create(const std::string& info) const = 0;

    virtual void destroy(Model* model) const = 0;

};

} // namespace gbkfit

#endif // GBKFIT_MODEL_HPP
