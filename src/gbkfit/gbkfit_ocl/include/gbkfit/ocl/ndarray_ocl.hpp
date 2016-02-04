#pragma once
#ifndef GBKFIT_OCL_NDARRAY_OCL_HPP
#define GBKFIT_OCL_NDARRAY_OCL_HPP

#include "gbkfit/ndarray.hpp"

namespace gbkfit {
namespace ocl {

class ndarray_ocl
{
};

class ndarray_device : public ndarray_ocl
{
};

class ndarray_host_pinned : public ndarray_ocl
{
};

class ndarray_host_wrapped : public ndarray_ocl
{
};

class ndarray_ocl_svm_cg : public ndarray_ocl
{
};

class ndarray_ocl_svm_fg : public ndarray_ocl
{
};



} // namespace ocl
} // namespace gbkfit

#endif // GBKFIT_OCL_NDARRAY_OCL_HPP
