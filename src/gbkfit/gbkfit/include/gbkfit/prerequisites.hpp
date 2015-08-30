#pragma once
#ifndef GBKFIT_PREREQUISITES_HPP
#define GBKFIT_PREREQUISITES_HPP

#include <cinttypes>
#include <cstring>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <system_error>
#include <valarray>
#include <vector>

#include <boost/current_function.hpp>

namespace gbkfit
{

class core;

class fitter;
class fitter_factory;

class model;
class model_factory;

class ndarray;
class ndarray_host;

class nddataset;

class parameters_fit_info;

} // namespace gbkfit

#endif // GBKFIT_PREREQUISITES_HPP
