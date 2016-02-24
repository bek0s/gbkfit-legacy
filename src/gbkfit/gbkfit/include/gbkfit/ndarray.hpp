#pragma once
#ifndef GBKFIT_NDARRAY_HPP
#define GBKFIT_NDARRAY_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

class NDArray
{

public:

    typedef float             value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef value_type&       reference;
    typedef const value_type& const_reference;
    typedef std::size_t       size_type;

private:

    NDShape m_shape;

public:

    virtual ~NDArray();

    const NDShape& get_shape(void) const;

    size_type get_size(void) const;

    size_type get_size_in_bytes(void) const;

    virtual void read_data(pointer dst) const = 0;

    virtual void write_data(const_pointer src) = 0;

    virtual void write_data(const NDArray* src) = 0;

    virtual pointer map(void) = 0;

    virtual void unmap(void) = 0;

protected:

    NDArray(const NDShape& shape);

};

} // namespace gbkfit

#endif // GBKFIT_NDARRAY_HPP
