#pragma once
#ifndef GBKFIT_NDSHAPE_HPP
#define GBKFIT_NDSHAPE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class NDShape
{

public:

    typedef int value_type;
    typedef std::vector<value_type>::size_type size_type;

private:

    std::vector<value_type> m_shape;

public:

    NDShape(void)
        : NDShape({}) {}

    template<typename Iterator>
    NDShape(Iterator shape_iter_first, Iterator shape_iter_last)
        : m_shape(shape_iter_first, shape_iter_last) {}

    template<typename T>
    NDShape(const std::vector<T>& shape)
        : NDShape(shape.begin(), shape.end()) {}

    template<typename T>
    NDShape(const std::initializer_list<T>& shape)
        : NDShape(shape.begin(), shape.end()) {}

    virtual ~NDShape() {}

    size_type get_dim_count(void) const;

    value_type get_dim_length(size_type idx) const;

    value_type get_dim_length_product(void) const;

    const std::vector<value_type>& as_vector(void) const;

    const value_type& operator[](size_type idx) const;

    value_type& operator[](size_type idx);

    bool operator==(const NDShape& rhs) const;

    bool operator!=(const NDShape& rhs) const;

}; // class NDShape

} // namespace gbkfit

#endif // GBKFIT_NDSHAPE_HPP
