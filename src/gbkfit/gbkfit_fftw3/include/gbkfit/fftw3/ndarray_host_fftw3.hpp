#pragma once
#ifndef GBKFIT_FFTW3_NDARRAY_HOST_FFTW3_HPP
#define GBKFIT_FFTW3_NDARRAY_HOST_FFTW3_HPP

#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {
namespace fftw3 {

//!
//! \brief The ndarray_host_fftw3 class
//!
class ndarray_host_fftw3 : public ndarray_host
{

public:

    ndarray_host_fftw3(const ndshape& shape);

    ndarray_host_fftw3(const ndshape& shape, const_pointer data);

    ~ndarray_host_fftw3();

}; // class ndarray_host_fftw3

} // namespace fftw3
} // namespace gbkfit

#endif // GBKFIT_FFTW3_NDARRAY_HOST_FFTW3_HPP
