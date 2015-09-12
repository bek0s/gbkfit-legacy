#pragma once
#ifndef GBKFIT_CONVOLVER_HPP
#define GBKFIT_CONVOLVER_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"
#include <tuple>

namespace gbkfit {

//!
//! \brief The convolver class
//!
class convolver
{

private:

    //ndshape m_shape;

public:

    convolver() {}

    virtual ~convolver() {}

    virtual void convolve(ndarray* data) = 0;

}; // class convolver

} // namespace gbkfit

#endif // GBKFIT_CONVOLVER_HPP
