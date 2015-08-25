#pragma once
#ifndef GBKFIT_NDARRAY_HOST_HPP
#define GBKFIT_NDARRAY_HOST_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndarray.hpp"

namespace gbkfit {


class ndarray_host : public ndarray
{

private:

    memory_buffer* m_memory_buffer;

public:

    ndarray_host(const ndshape &shape);

    ndarray_host(const ndshape& shape, const_pointer data);

    ~ndarray_host();

    pointer get_host_ptr(void);

    const_pointer get_host_ptr(void) const;

    void read_data(pointer dst) const final;

    void write_data(const_pointer data) final;

    void copy_data(const ndarray* src) final;

    void copy_data(const ndarray_host* src);

}; // class ndarray_host


} // namespace gbkfit

#endif // GBKFIT_NDARRAY_HOST_HPP
