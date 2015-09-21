#pragma once
#ifndef GBKFIT_PREREQUISITES_HPP
#define GBKFIT_PREREQUISITES_HPP

#include <cinttypes>
#include <cstring>

#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <system_error>
#include <valarray>
#include <vector>
#include <numeric>

#include <boost/current_function.hpp>

namespace gbkfit
{

class core;

class fitter;
class fitter_factory;

class instrument;

class model;
class model_factory;

class ndarray;
class ndarray_host;

class nddataset;

class parameters_fit_info;

class line_spread_function;
class point_spread_function;

} // namespace gbkfit

#endif // GBKFIT_PREREQUISITES_HPP
