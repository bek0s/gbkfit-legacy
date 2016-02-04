#pragma once
#ifndef GBKFIT_NDARRAY_HOST_HPP
#define GBKFIT_NDARRAY_HOST_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {

//!
//! \brief The ndarray_host class
//!
class NDArrayHost : public NDArray
{

protected:

    pointer m_data;

public:

    NDArrayHost(const NDShape &shape);

    ~NDArrayHost();

    pointer get_host_ptr(void);

    const_pointer get_host_ptr(void) const;

    void read_data(pointer dst) const final;

    void write_data(const_pointer data) final;

    void write_data(const NDArray* src) final;

    void write_data(const NDArrayHost* src);

}; // class ndarray_host

} // namespace gbkfit

#endif // GBKFIT_NDARRAY_HOST_HPP
