#pragma once
#ifndef GBKFIT_NDSHAPE_HPP
#define GBKFIT_NDSHAPE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


//!
//! \brief The ndshape class
//!
//! \todo Validate sizes during construction.
//! \todo Validate index in operator[].
//!
class ndshape
{

public:

    typedef typename std::vector<std::size_t>::size_type size_type;

private:

    std::vector<size_type> m_shape;

public:

    ndshape(const std::vector<size_type>& shape);

    ndshape(const std::initializer_list<size_type>& shape);

    virtual ~ndshape();

    size_type get_dim_count(void) const;

    size_type get_dim_length(size_type idx) const;

    size_type get_dim_length_product(void) const;

    const size_type& operator[](size_type idx) const;

    size_type& operator[](size_type idx);

    bool operator==(const ndshape& rhs) const;

    bool operator!=(const ndshape& rhs) const;

}; // class ndshape


} // namespace gbkfit

#endif // GBKFIT_NDSHAPE_HPP
