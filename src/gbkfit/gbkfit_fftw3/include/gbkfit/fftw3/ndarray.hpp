#pragma once
#ifndef GBKFIT_FFTW3_NDARRAY_HPP
#define GBKFIT_FFTW3_NDARRAY_HPP

#include "gbkfit/ndarray.hpp"

namespace gbkfit {
namespace fftw3 {

class NDArray : public gbkfit::NDArray
{

private:

    pointer m_data;

public:

    NDArray(const NDShape& shape);

    ~NDArray();

    pointer get_host_ptr(void);

    const_pointer get_host_ptr(void) const;

    void read_data(pointer dst) const override final;

    void write_data(const_pointer data) override final;

    void write_data(const gbkfit::NDArray* src) override final;

    pointer map(void) override final;

    void unmap(void) override final;

};

} // namespace fftw3
} // namespace gbkfit

#endif // GBKFIT_FFTW3_NDARRAY_HPP
