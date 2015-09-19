
#include "gbkfit/models/model01/model_model01_cuda.hpp"
#include "gbkfit/models/model01/kernels_cuda_host.hpp"
#include "gbkfit/cuda/ndarray_cuda.hpp"
#include "gbkfit/instrument.hpp"

namespace gbkfit {
namespace models {
namespace model01 {

model_model01_cuda::model_model01_cuda(profile_flx_type profile_flx, profile_vel_type profile_vel)
    : model_model01(profile_flx,profile_vel)
{
}

model_model01_cuda::~model_model01_cuda()
{
    delete m_data_velmap;
    delete m_data_sigmap;
}

void model_model01_cuda::initialize(int size_x, int size_y, int size_z, instrument* instrument)
{
    (void)size_x;
    (void)size_y;
    (void)size_z;
    (void)instrument;

    m_upsampling_x = 1;
    m_upsampling_y = 1;
    m_upsampling_z = 1;

    //
    // Store instrument information.
    //

    m_step_x = instrument->get_step_x();
    m_step_y = instrument->get_step_y();
    m_step_z = instrument->get_step_z();
    m_step_u_x = m_step_x*m_upsampling_x;
    m_step_u_y = m_step_y*m_upsampling_y;
    m_step_u_z = m_step_z*m_upsampling_z;

    /*
    m_data_psfcube = instrument->create_psf_cube_data(m_step_x, m_step_y, m_step_z);
    m_data_psfcube_u = instrument->create_psf_cube_data(m_step_u_x, m_step_u_y, m_step_u_z);
    m_psf_size_u_x = m_data_psfcube_u->get_shape()[0];
    m_psf_size_u_y = m_data_psfcube_u->get_shape()[1];
    m_psf_size_u_z = m_data_psfcube_u->get_shape()[2];
    */
}

const std::string& model_model01_cuda::get_type_name(void) const
{
    return model_factory_model01_cuda::FACTORY_TYPE_NAME;
}

const std::map<std::string,ndarray*>& model_model01_cuda::get_data(void) const
{
    return m_data_map;
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
