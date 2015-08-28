#pragma once
#ifndef GBKFIT_NDARRAY_HOST_HPP
#define GBKFIT_NDARRAY_HOST_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {


//!
//! \brief The ndarray_host class
//!
class ndarray_host : public ndarray
{

protected:

    memory_buffer* m_memory_buffer;

    value_type* m_data;

public:

    ~ndarray_host();

    pointer get_host_ptr(void);

    const_pointer get_host_ptr(void) const;

    void read_data(pointer dst) const final;

    void write_data(const_pointer data) final;

    void copy_data(const ndarray* src) final;

    void copy_data(const ndarray_host* src);

protected:

    ndarray_host(const ndshape &shape);

}; // class ndarray_host


//!
//! \brief The ndarray_host_malloc class
//!
class ndarray_host_malloc : public ndarray_host
{

public:

    ndarray_host_malloc(const ndshape& shape);

    ndarray_host_malloc(const ndshape& shape, const_pointer data);

    ~ndarray_host_malloc();

}; // class ndarray_host_malloc


//!
//! \brief The ndarray_host_new class
//!
class ndarray_host_new : public ndarray_host
{

public:

    ndarray_host_new(const ndshape& shape);

    ndarray_host_new(const ndshape& shape, const_pointer data);

    ~ndarray_host_new();

}; // class ndarray_host_new


} // namespace gbkfit

#endif // GBKFIT_NDARRAY_HOST_HPP
