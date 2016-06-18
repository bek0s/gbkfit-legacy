
#include "gbkfit/gmodel/gmodel1/gmodel1_omp.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_omp_factory.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_omp_kernels.hpp"

#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {

GModel1Omp::GModel1Omp(FlxProfileType flx_profile, VelProfileType vel_profile)
    : GModel1(flx_profile, vel_profile)
{
}

const std::string& GModel1Omp::get_type(void) const
{
    return GModel1OmpFactory::FACTORY_TYPE;
}

void GModel1Omp::evaluate(const std::vector<float>& params_prj,
                           const std::vector<float>& params_flx,
                           const std::vector<float>& params_vel,
                           float param_vsys,
                           float param_vsig,
                           const std::vector<float>& data_zero,
                           const std::vector<float>& data_step,
                           NDArray* data) const
{
    // Make sure the data storage is located on the host
    NDArrayHost* host_data = dynamic_cast<NDArrayHost*>(data);
    if (!host_data) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    // All good, evaluate model on the supplied storage
    NDShape data_shape = data->get_shape();
    kernels_omp::evaluate_model(m_flx_profile,
                                m_vel_profile,
                                params_prj.data(),
                                params_flx.data(),
                                params_vel.data(),
                                param_vsys,
                                param_vsig,
                                data_shape[0],
                                data_shape[1],
                                data_shape[2],
                                data_zero[0],
                                data_zero[1],
                                data_zero[2],
                                data_step[0],
                                data_step[1],
                                data_step[2],
                                host_data->get_host_ptr());
}

} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit
