
#include "gbkfit/models/galaxy_2d_omp/model_galaxy_2d_omp.hpp"
#include "gbkfit/models/galaxy_2d_omp/kernels_omp.hpp"
#include "gbkfit/ndarray_host.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace models {
namespace galaxy_2d {


model_galaxy_2d_omp::model_galaxy_2d_omp(int width, int height, float step_x, float step_y, int upsampling_x, int upsampling_y,
                                         profile_flux_type profile_flux, profile_rcur_type profile_rcur)
    : model_galaxy_2d(width,height,step_x,step_y,upsampling_x,upsampling_y,profile_flux,profile_rcur)
{
    // psf and lsf

    // cube

    // output maps
    m_model_flxmap = new ndarray_host_new({m_model_size_x,m_model_size_y});
    m_model_velmap = new ndarray_host_new({m_model_size_x,m_model_size_y});
    m_model_sigmap = new ndarray_host_new({m_model_size_x,m_model_size_y});
    m_model_data_list["flxmap"] = m_model_flxmap;
    m_model_data_list["velmap"] = m_model_velmap;
    m_model_data_list["sigmap"] = m_model_sigmap;

    m_parameter_names = {"xo","yo","pa","incl","i0","r0","rt","vt","vsys"};

}

model_galaxy_2d_omp::~model_galaxy_2d_omp()
{
    delete m_model_velmap;
    delete m_model_sigmap;
}

const std::string& model_galaxy_2d_omp::get_type_name(void) const
{
    return model_factory_galaxy_2d_omp::FACTORY_TYPE_NAME;
}

const std::map<std::string,ndarray*>& model_galaxy_2d_omp::get_data(void) const
{
    return m_model_data_list;
}

const std::map<std::string,ndarray*>& model_galaxy_2d_omp::evaluate(int model_flux_id,
                                                                    int model_rcur_id,
                                                                    const float parameter_vsys,
                                                                    const std::vector<float>& parameters_proj,
                                                                    const std::vector<float>& parameters_flux,
                                                                    const std::vector<float>& parameters_rcur,
                                                                    const std::vector<float>& parameters_vsig)
{
    // get native pointers for convenience access to the underlying data
    ndarray_host* model_flxmap = reinterpret_cast<ndarray_host*>(m_model_flxmap);
    ndarray_host* model_velmap = reinterpret_cast<ndarray_host*>(m_model_velmap);
    ndarray_host* model_sigmap = reinterpret_cast<ndarray_host*>(m_model_sigmap);

    // without psf
    if(true)
    {
        kernels_omp::model_image_2d_evaluate(model_flxmap->get_host_ptr(),
                                             model_velmap->get_host_ptr(),
                                             model_sigmap->get_host_ptr(),
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
    }
    // with psf
    else
    {
    }

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

    // Create parameter map.
    std::map<std::string,float> parameters;

    // Iterate over...
    for(auto& info_ptree_child : info_ptree)
    {
        // ...parameters
        if(info_ptree_child.first == "parameter")
        {
            // Parse name and value.
            std::string parameter_name = info_ptree_child.second.get<std::string>("<xmlattr>.name");
            float parameter_value = info_ptree_child.second.get<float>("<xmlattr>.value");
            // Add them to the parameters map.
            parameters.emplace(parameter_name,parameter_value);

            std::cout << parameter_name << "=" << parameter_value << std::endl;
        }
    }


    model* new_model = new gbkfit::models::galaxy_2d::model_galaxy_2d_omp(17,17,1,1,1,1,
                                                                          gbkfit::models::galaxy_2d::model_galaxy_2d::exponential,
                                                                          gbkfit::models::galaxy_2d::model_galaxy_2d::arctan);

    // Evaluate/Initialize model with the supplied parameter values.
    new_model->evaluate(parameters);

    return new_model;
}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
