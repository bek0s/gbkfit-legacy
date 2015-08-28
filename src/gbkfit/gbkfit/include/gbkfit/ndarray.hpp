#pragma once
#ifndef GBKFIT_NDARRAY_HPP
#define GBKFIT_NDARRAY_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {


//!
//! \brief The ndarray class
//!
class ndarray
{

public:

    typedef float             value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef value_type&       reference;
    typedef const value_type& const_reference;

private:

    ndshape m_shape;

public:

    virtual ~ndarray();

    const ndshape &get_shape(void) const;

    virtual void read_data(pointer dst) const = 0;

    virtual void write_data(const_pointer src) = 0;

    virtual void copy_data(const ndarray* src) = 0;

protected:

    ndarray(const ndshape& shape);

}; // class ndarray


} // namespace gbkfit

#endif // GBKFIT_NDARRAY_HPP
