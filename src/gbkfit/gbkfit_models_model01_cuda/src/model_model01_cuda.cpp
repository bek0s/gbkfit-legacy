
#include "gbkfit/models/model01/model_model01_cuda.hpp"
#include "gbkfit/models/model01/kernels_cuda_host.hpp"
#include "gbkfit/cuda/ndarray_cuda.hpp"

namespace gbkfit {
namespace models {
namespace model01 {


model_model01_cuda::model_model01_cuda(profile_flx_type profile_flx, profile_vel_type profile_vel)
    : model_model01(profile_flx,profile_vel)
{
}

model_model01_cuda::~model_model01_cuda()
{
    delete m_model_velmap;
    delete m_model_sigmap;
}

void model_model01_cuda::initialize(int size_x, int size_y, int size_z, instrument* instrument)
{
    (void)size_x;
    (void)size_y;
    (void)size_z;
    (void)instrument;
}

const std::string& model_model01_cuda::get_type_name(void) const
{
    return model_factory_model01_cuda::FACTORY_TYPE_NAME;
}

const std::map<std::string,ndarray*>& model_model01_cuda::get_data(void) const
{
    return m_model_data_list;
}

const std::map<std::string,ndarray*>& model_model01_cuda::evaluate(int profile_flx_id,
                                                                   int profile_vel_id,
                                                                   const std::vector<float>& params_prj,
                                                                   const std::vector<float>& params_flx,
                                                                   const std::vector<float>& params_vel,
                                                                   const float param_vsys,
                                                                   const float param_vsig)
{
    (void)profile_flx_id;
    (void)profile_vel_id;
    (void)params_prj;
    (void)params_flx;
    (void)params_vel;
    (void)param_vsys;
    (void)param_vsig;
    return get_data();
}

const std::string model_factory_model01_cuda::FACTORY_TYPE_NAME = "gbkfit.models.model_galaxy_2d_cuda";

model_factory_model01_cuda::model_factory_model01_cuda(void)
{
}

model_factory_model01_cuda::~model_factory_model01_cuda()
{
}

const std::string& model_factory_model01_cuda::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

model* model_factory_model01_cuda::create_model(const std::string& info) const
{
    (void)info;
    return new gbkfit::models::model01::model_model01_cuda(gbkfit::models::model01::exponential,
                                                           gbkfit::models::model01::arctan);
}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
