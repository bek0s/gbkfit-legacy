#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_HPP
#define GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_HPP

#include <cufft.h>

#include "gbkfit/dmodel/scube/scube.hpp"

namespace gbkfit {
namespace cuda {

class NDArrayDevice;
class NDArrayManaged;

} // namespace cuda
} // namespace gbkfit

namespace gbkfit {
namespace dmodel {
namespace scube {

class SCubeCuda : public SCube
{

private:

    std::vector<int> m_upsampling;

    std::vector<int> m_flxcube_size;
    std::vector<int> m_flxcube_size_u;
    std::vector<int> m_psfcube_size;
    std::vector<int> m_psfcube_size_u;
    std::vector<int> m_size_up;

    std::vector<float> m_step;
    std::vector<float> m_step_u;

    NDArrayHost* m_h_flxcube;
    NDArrayHost* m_h_flxcube_up;

    NDArrayHost* m_h_psfcube;
    NDArrayHost* m_h_psfcube_u;
    NDArrayHost* m_h_psfcube_up;

    cuda::NDArrayManaged* m_d_flxcube;
    cuda::NDArrayManaged* m_d_flxcube_up;
    cuda::NDArrayManaged* m_d_flxcube_up_fft;

    cuda::NDArrayManaged* m_d_psfcube;
    cuda::NDArrayManaged* m_d_psfcube_u;
    cuda::NDArrayManaged* m_d_psfcube_up;
    cuda::NDArrayManaged* m_d_psfcube_up_fft;

    std::map<std::string, NDArrayHost*> m_h_output_map;
    std::map<std::string, cuda::NDArrayManaged*> m_d_output_map;

    cufftHandle m_fft_plan_flxcube_r2c;
    cufftHandle m_fft_plan_flxcube_c2r;
    cufftHandle m_fft_plan_psfcube_r2c;

public:

    SCubeCuda(int size_x,
              int size_y,
              int size_z,
              float step_x,
              float step_y,
              float step_z,
              int upsampling_x,
              int upsampling_y,
              int upsampling_z,
              const PointSpreadFunction* psf,
              const LineSpreadFunction* lsf);

    ~SCubeCuda();

    const std::string& get_type(void) const override final;

    const std::vector<int>& get_size(void) const override final;

    const std::vector<float>& get_step(void) const override final;

    const std::map<std::string, NDArrayHost*>& evaluate(
            const std::map<std::string, float>& params) const override final;

    const std::map<std::string, cuda::NDArrayManaged*>& evaluate_managed(
            const std::map<std::string, float> &params) const;

};

} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_HPP
