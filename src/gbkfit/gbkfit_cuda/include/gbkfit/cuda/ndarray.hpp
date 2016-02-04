#pragma once
#ifndef GBKFIT_CUDA_NDARRAY_HPP
#define GBKFIT_CUDA_NDARRAY_HPP

#include "gbkfit/ndarray.hpp"

namespace gbkfit {
namespace cuda {

//!
//! \brief The gbkfit::cuda::ndarray class
//!
class NDArray : public gbkfit::NDArray
{

protected:

    pointer m_data;

public:

    ~NDArray();

    pointer get_cuda_ptr(void);

    const_pointer get_cuda_ptr(void) const;

    void read_data(pointer dst) const final;

    void write_data(const_pointer src) final;

    void write_data(const gbkfit::NDArray* src) final;

    void write_data(const gbkfit::cuda::NDArray* src);

protected:

    NDArray(const NDShape& shape);

}; // class ndarray

//!
//! \brief The ndarray_device class
//!
class ndarray_device : public gbkfit::cuda::NDArray
{

public:

    ndarray_device(const NDShape& shape);

    ~ndarray_device();

}; // class ndarray_device

//!
//! \brief The ndarray_pinned class
//!
class ndarray_pinned : public gbkfit::cuda::NDArray
{

public:

    ndarray_pinned(const NDShape& shape);

    ~ndarray_pinned();

}; // class ndarray_pinned

//!
//! \brief The ndarray_managed class
//!
class ndarray_managed : public gbkfit::cuda::NDArray
{

public:

    ndarray_managed(const NDShape& shape);

    ~ndarray_managed();

}; // class ndarray_managed

} // namespace cuda
} // namespace gbkfit

#endif // GBKFIT_CUDA_NDARRAY_HPP
