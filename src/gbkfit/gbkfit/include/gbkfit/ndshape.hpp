#pragma once
#ifndef GBKFIT_NDSHAPE_HPP
#define GBKFIT_NDSHAPE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


//!
//! \brief The ndshape class
//!
class ndshape
{

public:

    typedef int value_type;
    typedef std::vector<value_type>::size_type size_type;

private:

    std::vector<value_type> m_shape;

public:

    ndshape(const std::vector<value_type>& shape);

    ndshape(const std::initializer_list<value_type>& shape);

    virtual ~ndshape();

    size_type get_dim_count(void) const;

    value_type get_dim_length(size_type idx) const;

    value_type get_dim_length_product(void) const;

    const value_type& operator[](size_type idx) const;

    value_type& operator[](size_type idx);

    bool operator==(const ndshape& rhs) const;

    bool operator!=(const ndshape& rhs) const;

}; // class ndshape


} // namespace gbkfit

#endif // GBKFIT_NDSHAPE_HPP
