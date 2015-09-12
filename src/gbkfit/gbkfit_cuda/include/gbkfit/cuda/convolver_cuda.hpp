#pragma once
#ifndef GBKFIT_CUDA_CONVOLVER_HPP
#define GBKFIT_CUDA_CONVOLVER_HPP

#include "gbkfit/convolver.hpp"
#include <cufft.h>

namespace gbkfit {
namespace cuda {

class convolver_cuda : convolver
{

public:

    cufftHandle m_plan_kernel_r2c;

    cufftHandle m_plan_cube_r2c;
    cufftHandle m_plan_cube_c2r;

public:

    convolver_cuda(const ndshape& shape, const ndarray* psf, const ndarray* lsf);

    void convolve(ndarray* data);

};

/*
class model_cube : model
{
    ndarray* evaluate(params)
    {
        model->evaluate();

        // convolve (ydata, instrument)

        return ydata;
    }

};

class model_moments : model
{
    model_cube* m_model_cube;

    ndarray* evaluate(params)
    {
        auto data = m_model_cube->evaluate(params);

        m_data = convert_to_moments(data);

        return m_data;
    }
};
*/

} // namespace cuda
} // namespace gbkfit

#endif // GBKFIT_CUDA_CONVOLVER_HPP
