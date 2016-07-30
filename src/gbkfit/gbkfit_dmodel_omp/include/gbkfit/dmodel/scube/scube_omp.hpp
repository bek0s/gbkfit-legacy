#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_OMP_HPP
#define GBKFIT_DMODEL_SCUBE_SCUBE_OMP_HPP

#include "gbkfit/dmodel/scube/scube.hpp"

#include <fftw3.h>

namespace gbkfit {
namespace dmodel {
namespace scube {

class SCubeOmp : public SCube
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

    NDArrayHost* m_flxcube;
    NDArrayHost* m_flxcube_up;
    NDArrayHost* m_flxcube_up_fft;

    NDArrayHost* m_psfcube;
    NDArrayHost* m_psfcube_u;
    NDArrayHost* m_psfcube_up;
    NDArrayHost* m_psfcube_up_fft;

    std::map<std::string, NDArrayHost*> m_output_map;

    fftwf_plan m_fft_plan_flxcube_r2c;
    fftwf_plan m_fft_plan_flxcube_c2r;
    fftwf_plan m_fft_plan_psfcube_r2c;

public:

    SCubeOmp(int size_x,
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

    ~SCubeOmp();

    const std::string& get_type(void) const override final;

    const std::vector<int>& get_size(void) const override final;

    const std::vector<float>& get_step(void) const override final;

    const std::map<std::string, NDArrayHost*>& evaluate(
            const std::map<std::string, float>& params) const override final;

};

} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_OMP_HPP
