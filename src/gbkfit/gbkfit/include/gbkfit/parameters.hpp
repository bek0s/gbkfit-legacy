#pragma once
#ifndef GBKFIT_PARAMETERS_HPP
#define GBKFIT_PARAMETERS_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/property_map.hpp"

#include <boost/lexical_cast.hpp>

namespace gbkfit {

//!
//! \brief The Parameters class
//!
class Parameters
{

private:

    class Parameter : public PropertyMap
    {
    }; // class Parameter

private:

    std::map<std::string, Parameter> m_parameters;

public:

    Parameter& add_parameter(const std::string& name);

    Parameter& get_parameter(const std::string& name);



}; // class Parameters

} // namespace gbkfit

#endif // GBKFIT_PARAMETERS_HPP
