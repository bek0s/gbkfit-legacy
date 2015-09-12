#pragma once
#ifndef GBKFIT_GALMODEL_HPP
#define GBKFIT_GALMODEL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class galmodel
{

public:

    galmodel(void);

    virtual ~galmodel();

    virtual const std::string& get_type_name(void) const = 0;

    virtual const std::vector<std::string>& get_parameter_names(void) const = 0;

    virtual void evaluate(const std::map<std::string,float>& parameters, const float* data_step, ndarray* data) = 0;

}; // class galmodel

class galmodel_factory
{

public:

    galmodel_factory(void);

    virtual ~galmodel_factory();

    virtual const std::string& get_type_name(void) const = 0;

    virtual galmodel* create_galmodel(const std::string& info) const = 0;

}; // class galmodel_factory

} // namespace gbkfit

#endif // GBKFIT_GALMODEL_HPP
