#pragma once
#ifndef GBKFIT_PREREQUISITES_HPP
#define GBKFIT_PREREQUISITES_HPP

#include <cinttypes>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <map>
#include <system_error>

namespace gbkfit
{

class core;

class fitter;
class fitter_factory;

class memory_buffer;
class memory_buffer_host;

class model;
class model_factory;

class model_parameter_fit_info;
class model_parameters_fit_info;

class ndarray;
class nddataset;

} // namespace gbkfit

#endif // GBKFIT_PREREQUISITES_HPP
