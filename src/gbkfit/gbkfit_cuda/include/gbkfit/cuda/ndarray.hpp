#pragma once
#ifndef GBKFIT_CUDA_NDARRAY_HPP
#define GBKFIT_CUDA_NDARRAY_HPP

#include "gbkfit/ndarray.hpp"

namespace gbkfit {
namespace cuda {

class NDArray : public gbkfit::NDArray
{

protected:

    pointer m_data;

public:

    ~NDArray();

    pointer get_cuda_ptr(void);

    const_pointer get_cuda_ptr(void) const;

    void read_data(pointer dst) const override final;

    void write_data(const_pointer src) override final;

    void write_data(const gbkfit::NDArray* src) override final;

    pointer map(void) override = 0;

    void unmap(void) override = 0;

protected:

    NDArray(const NDShape& shape);

};

class NDArrayDevice : public gbkfit::cuda::NDArray
{

private:

    pointer m_data_mapped;

public:

    NDArrayDevice(const NDShape& shape);

    ~NDArrayDevice();

    pointer map(void) override final;

    void unmap(void) override final;

};

class NDArrayPinned : public gbkfit::cuda::NDArray
{

public:

    NDArrayPinned(const NDShape& shape);

    ~NDArrayPinned();

    pointer map(void) override final;

    void unmap(void) override final;

};

class NDArrayManaged : public gbkfit::cuda::NDArray
{

public:

    NDArrayManaged(const NDShape& shape);

    ~NDArrayManaged();

    pointer map(void) override final;

    void unmap(void) override final;

};

} // namespace cuda
} // namespace gbkfit

#endif // GBKFIT_CUDA_NDARRAY_HPP
