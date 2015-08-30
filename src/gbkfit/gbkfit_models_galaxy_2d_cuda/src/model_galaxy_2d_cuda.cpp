
#include "gbkfit/models/galaxy_2d_cuda/model_galaxy_2d_cuda.hpp"
#include "gbkfit/models/galaxy_2d_cuda/kernels_cuda_host.hpp"
#include "gbkfit/cuda/ndarray_cuda.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {


model_galaxy_2d_cuda::model_galaxy_2d_cuda(int width, int height, float step_x, float step_y, int upsampling_x, int upsampling_y,
                                         profile_flux_type profile_flux, profile_rcur_type profile_rcur)
    : model_galaxy_2d(width,height,step_x,step_y,upsampling_x,upsampling_y,profile_flux,profile_rcur)
{
    // psf and lsf

    // cube

    // output maps
    m_model_velmap = new gbkfit::cuda::ndarray_cuda_device({m_model_size_x,m_model_size_y});
    m_model_sigmap = new gbkfit::cuda::ndarray_cuda_device({m_model_size_x,m_model_size_y});
    m_model_data_list["velmap"] = m_model_velmap;
    m_model_data_list["sigmap"] = m_model_sigmap;

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

const std::map<std::string,ndarray*>& model_galaxy_2d_cuda::evaluate(int model_flux_id,
                                                                     int model_rcur_id,
                                                                     const float parameter_vsys,
                                                                     const std::vector<float>& parameters_proj,
                                                                     const std::vector<float>& parameters_flux,
                                                                     const std::vector<float>& parameters_rcur,
                                                                     const std::vector<float>& parameters_vsig)
{
    // get native pointers for convenience access to the underlying data
    cuda::ndarray_cuda* model_velmap = reinterpret_cast<cuda::ndarray_cuda*>(m_model_velmap);
    cuda::ndarray_cuda* model_sigmap = reinterpret_cast<cuda::ndarray_cuda*>(m_model_sigmap);

    // without psf
    if(true)
    {
        kernels_cuda_host::foo(model_velmap->get_cuda_ptr(),
                               model_velmap->get_cuda_ptr(),
                               m_model_size_x,
                               m_model_size_y);
        /*
        kernels_omp::model_image_2d_evaluate(model_velmap->get_cuda_ptr(),
                                             model_sigmap->get_cuda_ptr(),
                                             model_flux_id,
                                             model_rcur_id,
                                             m_model_size_x,
                                             m_model_size_y,
                                             m_step_x,
                                             m_step_y,
                                             parameter_vsys,
                                             parameters_proj.data(),
                                             parameters_proj.size(),
                                             parameters_flux.data(),
                                             parameters_flux.size(),
                                             parameters_rcur.data(),
                                             parameters_rcur.size(),
                                             parameters_vsig.data(),
                                             parameters_vsig.size());
                                             */
    }
    // with psf
    else
    {
    }

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
    return new gbkfit::models::galaxy_2d::model_galaxy_2d_cuda(17,17,1,1,1,1,
                                                               gbkfit::models::galaxy_2d::model_galaxy_2d::exponential,
                                                               gbkfit::models::galaxy_2d::model_galaxy_2d::arctan);
}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
