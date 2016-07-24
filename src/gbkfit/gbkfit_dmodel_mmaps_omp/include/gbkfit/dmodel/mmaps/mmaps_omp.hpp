#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_OMP_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_OMP_HPP

#include "gbkfit/dmodel/mmaps/mmaps.hpp"
#include "gbkfit/dmodel/scube/scube_omp.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

class MMapsOmp : public MMaps
{

private:

    scube::SCubeOmp* m_scube;

    NDArrayHost* m_flxmap;
    NDArrayHost* m_velmap;
    NDArrayHost* m_sigmap;

    std::map<std::string, NDArrayHost*> m_output_map;

public:

    MMapsOmp(int size_x,
             int size_y,
             float step_x,
             float step_y,
             const Instrument* instrument);

    MMapsOmp(int size_x,
             int size_y,
             float step_x,
             float step_y,
             int upsampling_x,
             int upsampling_y,
             const Instrument* instrument);

    ~MMapsOmp();

    const std::string& get_type(void) const override final;

    const std::vector<int>& get_size(void) const override final;

    const std::vector<float>& get_step(void) const override final;

    const GModel* get_galaxy_model(void) const override final;

    void set_galaxy_model(const GModel* galaxy_model) override final;

    const std::map<std::string, NDArrayHost*>& evaluate(
            const std::map<std::string, float>& params) const override final;

};

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_OMP_HPP
