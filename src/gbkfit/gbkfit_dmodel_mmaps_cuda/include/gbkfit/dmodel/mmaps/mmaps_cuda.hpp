#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_HPP

#include "gbkfit/dmodel/mmaps/mmaps.hpp"
#include "gbkfit/dmodel/scube/scube_cuda.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

class MMapsCuda : public MMaps
{

private:

    scube::SCubeCuda* m_scube;

    NDArrayHost* m_h_flxmap;
    NDArrayHost* m_h_velmap;
    NDArrayHost* m_h_sigmap;

    cuda::NDArrayManaged* m_d_flxmap;
    cuda::NDArrayManaged* m_d_velmap;
    cuda::NDArrayManaged* m_d_sigmap;

    std::map<std::string, NDArrayHost*> m_h_output_map;
    std::map<std::string, cuda::NDArrayManaged*> m_d_output_map;

public:

    MMapsCuda(int size_x,
              int size_y,
              const Instrument* instrument);

    MMapsCuda(int size_x,
              int size_y,
              int upsampling_x,
              int upsampling_y,
              const Instrument* instrument);

    ~MMapsCuda();

    const std::string& get_type(void) const;

    const Instrument* get_instrument(void) const override final;

    const GModel* get_galaxy_model(void) const override final;

    void set_galaxy_model(const GModel* gmodel) override final;

    const std::map<std::string, NDArrayHost*>& evaluate(
            const std::map<std::string, float>& params) const override final;

    const std::map<std::string, cuda::NDArrayManaged*>& evaluate_managed(
            const std::map<std::string, float>& params) const;

};

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_HPP