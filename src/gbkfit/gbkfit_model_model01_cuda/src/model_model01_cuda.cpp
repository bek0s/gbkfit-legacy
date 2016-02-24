
#include "gbkfit/models/model01/model_model01_cuda.hpp"
#include "gbkfit/models/model01/kernels_cuda_host.hpp"
#include "gbkfit/cuda/ndarray.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/utility.hpp"
#include "gbkfit/instrument.hpp"
#include <cufft.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
#include <cuda_runtime_api.h>

namespace gbkfit {
namespace models {
namespace model01 {

model_model01_cuda::model_model01_cuda(profile_flx_type profile_flx, profile_vel_type profile_vel)
    : ModelModel01(profile_flx,profile_vel)
{
}

model_model01_cuda::~model_model01_cuda()
{
    delete m_data_velmap;
    delete m_data_sigmap;
}

void model_model01_cuda::initialize(int size_x, int size_y, int size_z, Instrument* instrument)
{
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
    m_step_u_x = m_step_x/m_upsampling_x;
    m_step_u_y = m_step_y/m_upsampling_y;
    m_step_u_z = m_step_z/m_upsampling_z;

    NDArrayHost* data_psfcube = instrument->create_psf_cube_data(m_step_x, m_step_y, m_step_z);
    NDArrayHost* data_psfcube_u = instrument->create_psf_cube_data(m_step_u_x, m_step_u_y, m_step_u_z);
    m_data_psfcube = new cuda::NDArrayManaged(data_psfcube->get_shape());
    m_data_psfcube_u = new cuda::NDArrayManaged(data_psfcube_u->get_shape());
    m_data_psfcube->write_data(data_psfcube->get_host_ptr());
    m_data_psfcube_u->write_data(data_psfcube_u->get_host_ptr());
    m_psf_size_u_x = m_data_psfcube_u->get_shape()[0];
    m_psf_size_u_y = m_data_psfcube_u->get_shape()[1];
    m_psf_size_u_z = m_data_psfcube_u->get_shape()[2];

    //
    // Calculate sizes.
    //

    m_size_x = size_x;
    m_size_y = size_y;
    m_size_z = size_z;
    m_size_u_x = m_size_x * m_upsampling_x;
    m_size_u_y = m_size_y * m_upsampling_y;
    m_size_u_z = m_size_z * m_upsampling_z;
    m_size_up_x = m_size_u_x + m_psf_size_u_x - 1;
    m_size_up_y = m_size_u_y + m_psf_size_u_y - 1;
    m_size_up_z = m_size_u_z + m_psf_size_u_z - 1;
    m_size_up_x = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up_x), 512, 256);
    m_size_up_y = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up_y), 512, 256);
    m_size_up_z = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up_z), 512, 256);

    int size_padded     = m_size_up_z*m_size_up_y* m_size_up_x;
    int size_padded_fft = m_size_up_z*m_size_up_y*(m_size_up_x/2+1);

    //
    // Allocate memory for flux and psf cubes.
    //

    m_data_psfcube_up = new cuda::NDArrayManaged({m_size_up_x, m_size_up_y, m_size_up_z});
    m_data_flxcube_up = new cuda::NDArrayManaged({m_size_up_x, m_size_up_y, m_size_up_z});
    m_data_psfcube_up_fft = new cuda::NDArrayManaged({size_padded_fft*2});
    m_data_flxcube_up_fft = new cuda::NDArrayManaged({size_padded_fft*2});

    //
    // Create fft plans for flux and psf cubes.
    //

    cufftPlan3d(&m_fft_plan_flxcube_r2c,
                m_size_up_z,
                m_size_up_y,
                m_size_up_x,
                CUFFT_R2C);

    cufftPlan3d(&m_fft_plan_flxcube_c2r,
                m_size_up_z,
                m_size_up_y,
                m_size_up_x,
                CUFFT_C2R);

    cufftPlan3d(&m_fft_plan_psfcube_r2c,
                m_size_up_z,
                m_size_up_y,
                m_size_up_x,
                CUFFT_R2C);

    //
    // Prepare psf and fft-transform it.
    //

    util_image::image_copy_padded(m_data_psfcube_u->get_cuda_ptr(),
                                  m_psf_size_u_x,
                                  m_psf_size_u_y,
                                  m_psf_size_u_z,
                                  m_size_up_x,
                                  m_size_up_y,
                                  m_size_up_z,
                                  m_data_psfcube_up->get_cuda_ptr());

    util_image::image_fill_padded(0,
                                  m_psf_size_u_x,
                                  m_psf_size_u_y,
                                  m_psf_size_u_z,
                                  m_size_up_x,
                                  m_size_up_y,
                                  m_size_up_z,
                                  m_data_psfcube_up->get_cuda_ptr());

    util_image::image_shift(m_data_psfcube_up->get_cuda_ptr(),
                            m_size_up_x,
                            m_size_up_y,
                            m_size_up_z,
                           -m_psf_size_u_x/2,
                           -m_psf_size_u_y/2,
                           -m_psf_size_u_z/2);

    cufftExecR2C(m_fft_plan_psfcube_r2c,
                 m_data_psfcube_up->get_cuda_ptr(),
                 (cufftComplex*)m_data_psfcube_up_fft->get_cuda_ptr());
    cudaDeviceSynchronize();

    //
    // Allocate and initialize output arrays.
    //

    m_data_flxcube = new cuda::NDArrayManaged({size_x, size_y, size_z});
    m_data_flxmap  = new cuda::NDArrayManaged({size_x, size_y});
    m_data_velmap  = new cuda::NDArrayManaged({size_x, size_y});
    m_data_sigmap  = new cuda::NDArrayManaged({size_x, size_y});

    std::fill(m_data_flxcube->get_cuda_ptr(), m_data_flxcube->get_cuda_ptr() + size_x*size_y*size_z, -1);
    std::fill(m_data_flxmap ->get_cuda_ptr(), m_data_flxmap ->get_cuda_ptr() + size_x*size_y,        -1);
    std::fill(m_data_velmap ->get_cuda_ptr(), m_data_velmap ->get_cuda_ptr() + size_x*size_y,        -1);
    std::fill(m_data_sigmap ->get_cuda_ptr(), m_data_sigmap ->get_cuda_ptr() + size_x*size_y,        -1);

    //
    // Add output data to the output map.
    //

    m_data_map["flxcube_up"] = m_data_flxcube_up;
    m_data_map["psfcube_up"] = m_data_psfcube_up;
    m_data_map["psfcube_u"] = m_data_psfcube_u;
    m_data_map["psfcube"] = m_data_psfcube;
    m_data_map["flxcube"] = m_data_flxcube;
    m_data_map["flxmap"] = m_data_flxmap;
    m_data_map["velmap"] = m_data_velmap;
    m_data_map["sigmap"] = m_data_sigmap;

}

const std::string& model_model01_cuda::get_type_name(void) const
{
    return model_factory_model01_cuda::FACTORY_TYPE_NAME;
}

const std::map<std::string,NDArray*>& model_model01_cuda::get_data(void) const
{
    return m_data_map;
}

const std::map<std::string,NDArray*>& model_model01_cuda::evaluate(int profile_flx_id,
                                                                   int profile_vel_id,
                                                                   const std::vector<float>& params_prj,
                                                                   const std::vector<float>& params_flx,
                                                                   const std::vector<float>& params_vel,
                                                                   const float param_vsys,
                                                                   const float param_vsig)
{
    //
    // Calculate margins.
    //

    int padding_u_x0 = m_psf_size_u_x/2;
    int padding_u_y0 = m_psf_size_u_x/2;
    int padding_u_z0 = m_psf_size_u_x/2;
    int padding_u_x1 = m_psf_size_u_x - padding_u_x0 - 1;
    int padding_u_y1 = m_psf_size_u_x - padding_u_y0 - 1;
    int padding_u_z1 = m_psf_size_u_x - padding_u_z0 - 1;

    //
    // Set a constant value to all pixels. Used for debug.
    //

#if 0
    float fill_value = -0.01;
    kernels_cuda_host::array_3d_fill(m_size_up_x,
                                     m_size_up_y,
                                     m_size_up_z,
                                     fill_value,
                                     m_data_flxcube_up->get_cuda_ptr());
#endif

    //
    // Evaluate cube model (upsampling + padding).
    //

    kernels_cuda_host::model_image_3d_evaluate(profile_flx_id,
                                               profile_vel_id,
                                               param_vsig,
                                               param_vsys,
                                               params_prj.data(),
                                               params_flx.data(),
                                               params_vel.data(),
                                               m_size_u_x,
                                               m_size_u_y,
                                               m_size_u_z,
                                               m_size_up_x,
                                               m_size_up_y,
                                               m_size_up_z,
                                               m_step_u_x,
                                               m_step_u_y,
                                               m_step_u_z,
                                               padding_u_x0,
                                               padding_u_y0,
                                               padding_u_z0,
                                               padding_u_x1,
                                               padding_u_y1,
                                               padding_u_z1,
                                               m_data_flxcube_up->get_cuda_ptr());

    //
    // Perform fft-based 3d convolution with the psf cube.
    //

#if 1
    kernels_cuda_host::model_image_3d_convolve_fft(m_data_flxcube_up->get_cuda_ptr(),
                                                   reinterpret_cast<cufftComplex*>(m_data_flxcube_up_fft->get_cuda_ptr()),
                                                   reinterpret_cast<cufftComplex*>(m_data_psfcube_up_fft->get_cuda_ptr()),
                                                   m_size_up_x,
                                                   m_size_up_y,
                                                   m_size_up_z,
                                                   m_fft_plan_flxcube_r2c,
                                                   m_fft_plan_flxcube_c2r);
#endif

    //
    // Downsample cube and remove padding.
    //

    kernels_cuda_host::model_image_3d_downsample_and_copy(m_data_flxcube_up->get_cuda_ptr(),
                                                          m_size_x,
                                                          m_size_y,
                                                          m_size_z,
                                                          m_size_up_x,
                                                          m_size_up_y,
                                                          m_size_up_z,
                                                          padding_u_x0,
                                                          padding_u_y0,
                                                          padding_u_z0,
                                                          m_upsampling_x,
                                                          m_upsampling_y,
                                                          m_upsampling_z,
                                                          m_data_flxcube->get_cuda_ptr());

    //
    // Extract moment maps from cube.
    //

    kernels_cuda_host::model_image_3d_extract_moment_maps(m_data_flxcube->get_cuda_ptr(),
                                                          m_size_x,
                                                          m_size_y,
                                                          m_size_z,
                                                          m_step_x,
                                                          m_step_y,
                                                          m_step_z,
                                                          m_data_flxmap->get_cuda_ptr(),
                                                          m_data_velmap->get_cuda_ptr(),
                                                          m_data_sigmap->get_cuda_ptr());

    return get_data();
}

const std::string model_factory_model01_cuda::FACTORY_TYPE_NAME = "gbkfit.model.model_model01_cuda";

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

Model* model_factory_model01_cuda::create_model(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    std::string profile_flx_name = info_ptree.get<std::string>("profile_flx");
    std::string profile_vel_name = info_ptree.get<std::string>("profile_vel");

    profile_flx_type profile_flx;
    profile_vel_type profile_vel;

    if (profile_flx_name == "exponential")
        profile_flx = gbkfit::models::model01::exponential;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    if      (profile_vel_name == "lramp")
        profile_vel = gbkfit::models::model01::lramp;
    else if (profile_vel_name == "david")
        profile_vel = gbkfit::models::model01::david;
    else if (profile_vel_name == "arctan")
        profile_vel = gbkfit::models::model01::arctan;
    else if (profile_vel_name == "epinat")
        profile_vel = gbkfit::models::model01::epinat;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    return new gbkfit::models::model01::model_model01_cuda(profile_flx,profile_vel);
}

} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
