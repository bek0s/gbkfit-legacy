#pragma once
#ifndef GBKFIT_NDARRAY_HOST_HPP
#define GBKFIT_NDARRAY_HOST_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {

class NDArrayHost : public NDArray
{

protected:

    pointer m_data;

public:

    NDArrayHost(const NDShape& shape);

    NDArrayHost(const NDShape& shape, const value_type& value);

    ~NDArrayHost();

    pointer get_host_ptr(void);

    const_pointer get_host_ptr(void) const;

    void read_data(pointer dst) const override final;

    void write_data(const_pointer data) override final;

    void write_data(const NDArray* src) override final;

    pointer map(void) override final;

    void unmap(void) override final;

};

} // namespace gbkfit

#endif // GBKFIT_NDARRAY_HOST_HPP
