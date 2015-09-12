
#include "gbkfit/models/galaxy_2d_cuda/model_galaxy_2d_cuda.hpp"
#include "gbkfit/models/galaxy_2d_cuda/kernels_cuda_host.hpp"
#include "gbkfit/cuda/ndarray_cuda.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {


model_galaxy_2d_cuda::model_galaxy_2d_cuda(profile_flx_type profile_flx, profile_vel_type profile_vel)
    : model_galaxy_2d(profile_flx,profile_vel)
{
}

model_galaxy_2d_cuda::~model_galaxy_2d_cuda()
{
    delete m_model_velmap;
    delete m_model_sigmap;
}

const std::string& model_galaxy_2d_cuda::get_type_name(void) const
{
    return model_factory_galaxy_2d_cuda::FACTORY_TYPE_NAME;
}


const std::map<std::string,ndarray*>& model_galaxy_2d_cuda::get_data(void) const
{
    return m_model_data_list;
}

const std::map<std::string,ndarray*>& model_galaxy_2d_cuda::evaluate(int profile_flx_id,
                                                                     int profile_vel_id,
                                                                     const float param_vsig,
                                                                     const float param_vsys,
                                                                     const std::vector<float>& params_prj,
                                                                     const std::vector<float>& params_flx,
                                                                     const std::vector<float>& params_vel)
{
    return get_data();
}

const std::string model_factory_galaxy_2d_cuda::FACTORY_TYPE_NAME = "gbkfit.models.model_galaxy_2d_cuda";

model_factory_galaxy_2d_cuda::model_factory_galaxy_2d_cuda(void)
{
}

model_factory_galaxy_2d_cuda::~model_factory_galaxy_2d_cuda()
{
}

const std::string& model_factory_galaxy_2d_cuda::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

model* model_factory_galaxy_2d_cuda::create_model(const std::string& info) const
{
    return new gbkfit::models::galaxy_2d::model_galaxy_2d_cuda(gbkfit::models::galaxy_2d::exponential,
                                                               gbkfit::models::galaxy_2d::arctan);
}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
