#pragma once
#ifndef GBKFIT_NDSHAPE_HPP
#define GBKFIT_NDSHAPE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


//!
//! \brief The gbkfit::ndshape class
//!
class NDShape
{

public:

    typedef int value_type;
    typedef std::vector<value_type>::size_type size_type;

private:

    std::vector<value_type> m_shape;

public:

    NDShape(const std::vector<value_type>& shape);

    NDShape(const std::initializer_list<value_type>& shape);

    virtual ~NDShape();

    size_type get_dim_count(void) const;

    value_type get_dim_length(size_type idx) const;

    value_type get_dim_length_product(void) const;

    const std::vector<value_type>& get_as_vector(void) const;

    const value_type& operator[](size_type idx) const;

    value_type& operator[](size_type idx);

    bool operator==(const NDShape& rhs) const;

    bool operator!=(const NDShape& rhs) const;

}; // class ndshape


} // namespace gbkfit

#endif // GBKFIT_NDSHAPE_HPP
