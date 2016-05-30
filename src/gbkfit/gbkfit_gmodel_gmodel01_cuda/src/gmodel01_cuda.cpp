
#include "gbkfit/gmodel/gmodel01/gmodel01_cuda.hpp"
#include "gbkfit/gmodel/gmodel01/gmodel01_cuda_factory.hpp"
#include "gbkfit/gmodel/gmodel01/gmodel01_cuda_kernels_h.hpp"

#include "gbkfit/cuda/ndarray.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel01 {

GModel01Cuda::GModel01Cuda(FlxProfileType flx_profile,
                           VelProfileType vel_profile)
    : GModel01(flx_profile, vel_profile)
{
}

const std::string& GModel01Cuda::get_type(void) const
{
    return GModel01CudaFactory::FACTORY_TYPE;
}

void GModel01Cuda::evaluate(const std::vector<float>& params_prj,
                            const std::vector<float>& params_flx,
                            const std::vector<float>& params_vel,
                            float param_vsys,
                            float param_vsig,
                            const std::vector<float>& data_zero,
                            const std::vector<float>& data_step,
                            NDArray* data) const
{
    // Make sure the data storage is located on cuda managed memory
    cuda::NDArrayManaged* cuda_data = dynamic_cast<cuda::NDArrayManaged*>(data);
    if (!cuda_data) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    // Make sure the data storage has 3 dimensions
    NDShape data_shape = data->get_shape();
    kernels_cuda_h::evaluate_model(m_flx_profile,
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
                                   cuda_data->get_cuda_ptr());
}

} // namespace gmodel01
} // namespace gmodel
} // namespace gbkfit
