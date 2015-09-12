
#include "gbkfit/models/galaxy_2d_omp/model_galaxy_2d_omp.hpp"
#include "gbkfit/models/galaxy_2d_omp/kernels_omp.hpp"
#include "gbkfit/ndarray_host.hpp"

#include "gbkfit/instrument.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "gbkfit/utility.hpp"
namespace gbkfit {
namespace models {
namespace galaxy_2d {


model_galaxy_2d_omp::model_galaxy_2d_omp(profile_flx_type profile_flx, profile_vel_type profile_vel)
    : model_galaxy_2d(profile_flx,profile_vel)
{
}

model_galaxy_2d_omp::~model_galaxy_2d_omp()
{
    delete m_data_flxcube;
    delete m_data_flxcube_padded;
    delete m_data_flxmap;
    delete m_data_velmap;
    delete m_data_sigmap;
}


int data_padding_x;
int data_padding_y;
int data_padding_z;


void model_galaxy_2d_omp::initialize(int size_x, int size_y, int size_z, instrument *instrument)
{
    m_instrument = instrument;

    m_step_x = m_instrument->get_step_x();
    m_step_y = m_instrument->get_step_y();
    m_step_z = m_instrument->get_step_z();

    int psf_size_x = (10 * 4.0) / m_step_x;
    int psf_size_y = (10 * 4.0) / m_step_y;
    int psf_size_z = (10 * 4.0) / m_step_z;

    float* psf_data_spat = new float[psf_size_x*psf_size_y];
    float* psf_data_spec = new float[psf_size_z];


//  ndarray_host* psf_spat = new ndarray_host_new({psf_size_x,psf_size_y});
//  ndarray_host* psf_spec = new ndarray_host_new({psf_size_z});

    m_psf_size_x = 0;
    m_psf_size_y = 0;
    m_psf_size_z = 0;

    m_upsampling_x = 1;
    m_upsampling_y = 1;
    m_upsampling_z = 1;

    int size_x_padded = m_upsampling_x*size_x;
    int size_y_padded = m_upsampling_y*size_y;
    int size_z_padded = m_upsampling_z*size_z;

    data_padding_x = m_psf_size_x/2;
    data_padding_y = m_psf_size_y/2;
    data_padding_z = m_psf_size_z/2;


    if (m_psf_size_x > 0) size_x_padded += m_psf_size_x-1;
    if (m_psf_size_y > 0) size_y_padded += m_psf_size_y-1;
    if (m_psf_size_z > 0) size_z_padded += m_psf_size_z-1;

    size_x_padded = util_fft::calculate_optimal_dim_length(size_x_padded, 512, 256);
    size_y_padded = util_fft::calculate_optimal_dim_length(size_y_padded, 512, 256);
    size_z_padded = util_fft::calculate_optimal_dim_length(size_z_padded, 512, 256);

    m_data_flxcube = new ndarray_host_new({size_x,size_y,size_z});
    m_data_flxcube_padded = new ndarray_host_new({size_x_padded,size_y_padded,size_z_padded});
    m_data_flxmap = new ndarray_host_new({size_x,size_y});
    m_data_velmap = new ndarray_host_new({size_x,size_y});
    m_data_sigmap = new ndarray_host_new({size_x,size_y});
    m_data_map["flxcube"] = m_data_flxcube;
    m_data_map["flxmap"] = m_data_flxmap;
    m_data_map["velmap"] = m_data_velmap;
    m_data_map["sigmap"] = m_data_sigmap;
    m_data_map["flxcubea"] = m_data_flxcube_padded;

    std::cout << size_x_padded << std::endl
              << size_y_padded << std::endl
              << size_z_padded << std::endl;


}

const std::string& model_galaxy_2d_omp::get_type_name(void) const
{
    return model_factory_galaxy_2d_omp::FACTORY_TYPE_NAME;
}

const std::map<std::string,ndarray*>& model_galaxy_2d_omp::get_data(void) const
{
    return m_data_map;
}

const std::map<std::string,ndarray*>& model_galaxy_2d_omp::evaluate(int profile_flx_id,
                                                                    int profile_vel_id,
                                                                    const float param_vsig,
                                                                    const float param_vsys,
                                                                    const std::vector<float>& params_prj,
                                                                    const std::vector<float>& params_flx,
                                                                    const std::vector<float>& params_vel)
{
#if 0
    kernels_omp::model_image_3d_evaluate(profile_flx_id,
                                         profile_vel_id,
                                         param_vsig,
                                         param_vsys,
                                         params_prj.data(),
                                         params_flx.data(),
                                         params_vel.data(),
                                         m_data_flxcube->get_shape()[0],
                                         m_data_flxcube->get_shape()[1],
                                         m_data_flxcube->get_shape()[2],
                                         m_data_flxcube_padded->get_shape()[0],
                                         m_data_flxcube_padded->get_shape()[1],
                                         m_data_flxcube_padded->get_shape()[2],
                                         0,0,0,
                                         m_step_x,
                                         m_step_y,
                                         m_step_z,
                                         m_data_flxcube->get_host_ptr());
#else

    /*
    int data_padding_x = m_psf_size_x / 2;
    int data_padding_y = m_psf_size_y / 2;
    int data_padding_z = m_psf_size_z / 2;
    */

    kernels_omp::model_image_3d_evaluate(profile_flx_id,
                                         profile_vel_id,
                                         param_vsig,
                                         param_vsys,
                                         params_prj.data(),
                                         params_flx.data(),
                                         params_vel.data(),
                                         m_data_flxcube->get_shape()[0],
                                         m_data_flxcube->get_shape()[1],
                                         m_data_flxcube->get_shape()[2],
                                         m_data_flxcube_padded->get_shape()[0],
                                         m_data_flxcube_padded->get_shape()[1],
                                         m_data_flxcube_padded->get_shape()[2],
                                         data_padding_x,
                                         data_padding_y,
                                         data_padding_z,
                                         m_step_x,
                                         m_step_y,
                                         m_step_z,
                                         m_upsampling_x,
                                         m_upsampling_y,
                                         m_upsampling_z,
                                         m_data_flxcube_padded->get_host_ptr());


    kernels_omp::model_image_3d_downsample_and_copy(m_data_flxcube_padded->get_host_ptr(),
                                                    m_data_flxcube->get_shape()[0],
                                                    m_data_flxcube->get_shape()[1],
                                                    m_data_flxcube->get_shape()[2],
                                                    m_data_flxcube_padded->get_shape()[0],
                                                    m_data_flxcube_padded->get_shape()[1],
                                                    m_data_flxcube_padded->get_shape()[2],
                                                    data_padding_x,
                                                    data_padding_y,
                                                    data_padding_z,
                                                    m_upsampling_x,
                                                    m_upsampling_y,
                                                    m_upsampling_z,
                                                    m_data_flxcube->get_host_ptr());


    /*
    kernels_omp::model_image_3d_copy(m_data_flxcube_padded->get_host_ptr(),
                                     m_data_flxcube->get_shape()[0],
                                     m_data_flxcube->get_shape()[1],
                                     m_data_flxcube->get_shape()[2],
                                     m_data_flxcube_padded->get_shape()[0],
                                     m_data_flxcube_padded->get_shape()[1],
                                     m_data_flxcube_padded->get_shape()[2],
                                     m_data_flxcube->get_host_ptr());
                                     */

#endif


    kernels_omp::model_image_3d_extract_moment_maps(m_data_flxcube->get_host_ptr(),
                                                    m_data_flxcube->get_shape()[0],
                                                    m_data_flxcube->get_shape()[1],
                                                    m_data_flxcube->get_shape()[2],
                                                    m_step_x,
                                                    m_step_y,
                                                    m_step_z,
                                                    m_data_flxmap->get_host_ptr(),
                                                    m_data_velmap->get_host_ptr(),
                                                    m_data_sigmap->get_host_ptr());


    return get_data();
}

const std::string model_factory_galaxy_2d_omp::FACTORY_TYPE_NAME = "gbkfit.models.model_galaxy_2d_omp";

model_factory_galaxy_2d_omp::model_factory_galaxy_2d_omp(void)
{
}

model_factory_galaxy_2d_omp::~model_factory_galaxy_2d_omp()
{
}

const std::string& model_factory_galaxy_2d_omp::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

model* model_factory_galaxy_2d_omp::create_model(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Read flux and velocity profile names.
    std::string profile_flx_name = info_ptree.get<std::string>("profile_flx");
    std::string profile_vel_name = info_ptree.get<std::string>("profile_vel");

    profile_flx_type profile_flx;
    profile_vel_type profile_vel;

    if (profile_flx_name == "exponential")
        profile_flx = gbkfit::models::galaxy_2d::exponential;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    if (profile_vel_name == "lramp")
        profile_vel = gbkfit::models::galaxy_2d::lramp;
    else if (profile_vel_name == "arctan")
        profile_vel = gbkfit::models::galaxy_2d::arctan;
    else if (profile_vel_name == "epinat")
        profile_vel = gbkfit::models::galaxy_2d::epinat;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    return new gbkfit::models::galaxy_2d::model_galaxy_2d_omp(profile_flx,profile_vel);
}

} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
