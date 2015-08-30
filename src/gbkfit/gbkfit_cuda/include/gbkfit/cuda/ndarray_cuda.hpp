#pragma once
#ifndef GBKFIT_CUDA_NDARRAY_CUDA_HPP
#define GBKFIT_CUDA_NDARRAY_CUDA_HPP

#include "gbkfit/ndarray.hpp"

namespace gbkfit {
namespace cuda {

//!
//! \brief The ndarray_cuda class
//!
class ndarray_cuda : public ndarray
{

protected:

    pointer m_data;

public:

    ~ndarray_cuda();

    pointer get_cuda_ptr(void);

    const_pointer get_cuda_ptr(void) const;

    void read_data(pointer dst) const final;

    void write_data(const_pointer src) final;

    void copy_data(const ndarray* src) final;

    void copy_data(const ndarray_cuda* src);

protected:

    ndarray_cuda(const ndshape& shape);

}; // class ndarray_cuda

//!
//! \brief The ndarray_cuda_device class
//!
class ndarray_cuda_device : public ndarray_cuda
{

public:

    ndarray_cuda_device(const ndshape& shape);

    ndarray_cuda_device(const ndshape& shape, const_pointer data);

    ~ndarray_cuda_device();

}; // class ndarray_cuda_device

//!
//! \brief The ndarray_cuda_pinned class
//!
class ndarray_cuda_pinned : public ndarray_cuda
{

public:

    ndarray_cuda_pinned(const ndshape& shape);

    ndarray_cuda_pinned(const ndshape& shape, const_pointer data);

    ~ndarray_cuda_pinned();

}; // class ndarray_cuda_pinned

//!
//! \brief The ndarray_cuda_pinned_wc class
//!
class ndarray_cuda_pinned_wc : public ndarray_cuda
{

public:

    ndarray_cuda_pinned_wc(const ndshape& shape);

    ndarray_cuda_pinned_wc(const ndshape& shape, const_pointer data);

    ~ndarray_cuda_pinned_wc();

}; // class ndarray_cuda_pinned_wc

} // namespace cuda
} // namespace gbkfit

#endif // GBKFIT_CUDA_NDARRAY_CUDA_HPP
