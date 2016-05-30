#pragma once
#ifndef GBKFIT_PREREQUISITES_HPP
#define GBKFIT_PREREQUISITES_HPP

#include "gbkfit/config.hpp"

#include <cassert>
#include <cinttypes>
#include <cstring>

#include <algorithm>
#include <array>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <system_error>
#include <valarray>
#include <vector>

#include <boost/current_function.hpp>

namespace gbkfit
{

class Core;
class Data;
class DModel;
class DModelFactory;
class Dataset;
class Fitter;
class FitterFactory;
class FitterResult;
class FitterResultMode;
class GModel;
class GModelFactory;
class Instrument;
class Model;
class ModelFactory;
class NDArray;
class NDArrayHost;
class NDShape;
class Parameter;
class Parameters;

class LineSpreadFunction;
class PointSpreadFunction;

} // namespace gbkfit

#endif // GBKFIT_PREREQUISITES_HPP
