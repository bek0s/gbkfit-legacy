#pragma once
#ifndef GBKFIT_PREREQUISITES_HPP
#define GBKFIT_PREREQUISITES_HPP

#include "gbkfit/config.hpp"

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

#include <boost/current_function.hpp>

namespace gbkfit
{

class Core;

class Fitter;
class FitterFactory;

class instrument;

class Model;
class ModelFactory;

class NDArray;
class NDArrayHost;

class Dataset;

class Parameters;

class line_spread_function;
class point_spread_function;

} // namespace gbkfit

#endif // GBKFIT_PREREQUISITES_HPP
