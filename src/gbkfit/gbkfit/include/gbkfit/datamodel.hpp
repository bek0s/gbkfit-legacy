#pragma once
#ifndef GBKFIT_DATAMODEL_HPP
#define GBKFIT_DATAMODEL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class datamodel
{

public:

    datamodel(void);

    virtual ~datamodel();

    virtual const std::vector<std::string>& get_parameter_names(void) const = 0;

//  virtual const std::vector<float>& get_parameter_values(void) const = 0;

//  virtual const std::map<std::string,ndarray*>& get_data(void) const = 0;

    virtual const std::map<std::string,ndarray*>& evaluate(const std::map<std::string,float>& parameters) = 0;

}; // class datamodel

class datamodel_factory
{

public:

    datamodel_factory(void);

    virtual ~datamodel_factory();

    virtual const std::string& get_type_name(void) const = 0;

    virtual datamodel* create_datamodel(const std::string& info) const = 0;

}; // class datamodel_factory

} // namespace gbkfit

#endif // GBKFIT_DATAMODEL_HPP
