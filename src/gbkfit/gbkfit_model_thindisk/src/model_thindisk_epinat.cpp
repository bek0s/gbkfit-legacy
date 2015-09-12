
#include "gbkfit/model_thindisk/model_thindisk_epinat.hpp"

namespace gbkfit {
namespace model_thindisk {

const std::string MODEL_TYPE_NAME = "gbkfit.model_thindisk.epinat";

model_thindisk_epinat::model_thindisk_epinat(std::size_t width,
                                             std::size_t height,
                                             std::size_t depth,
                                             float step_x,
                                             float step_y,
                                             float step_z,
                                             std::size_t upsampling_x,
                                             std::size_t upsampling_y,
                                             std::size_t upsampling_z,
                                             const gbkfit::ndarray* psf)
    : model_thindisk(width,height,depth,step_x,step_y,step_z,upsampling_x,upsampling_y,upsampling_z,psf)
{
}

model_thindisk_epinat::~model_thindisk_epinat()
{
}

const std::string& model_thindisk_epinat::get_type_name(void) const
{
    return MODEL_TYPE_NAME;
}

std::vector<std::string> model_thindisk_epinat::get_parameter_names(void) const
{
    return std::vector<std::string>({"i0", "r0", "xo", "yo", "pa", "incl", "rt", "vt", "vsys", "vsig", "a", "g"});
}

void model_thindisk_epinat::evaluate(const std::map<std::string,float>& model_params, std::valarray<float>& model_data)
{
    // projection parameters
    std::vector<float> model_params_proj;
    model_params_proj.push_back(model_params.at("xo"));
    model_params_proj.push_back(model_params.at("yo"));
    model_params_proj.push_back(model_params.at("pa"));
    model_params_proj.push_back(model_params.at("incl"));

    // flux profile parameters
    std::vector<float> model_params_flux;
    model_params_flux.push_back(model_params.at("i0"));
    model_params_flux.push_back(model_params.at("r0"));

    // rotation curve parameters
    std::vector<float> model_params_rcur;
    model_params_rcur.push_back(model_params.at("rt"));
    model_params_rcur.push_back(model_params.at("vt"));
    model_params_rcur.push_back(model_params.at("a"));
    model_params_rcur.push_back(model_params.at("g"));

    // velocity dispersion parameters
    std::vector<float> model_params_vsig;
    model_params_vsig.push_back(model_params.at("vsig"));

    // evaluate model
    model_thindisk::evaluate(2,
                             model_params_proj,
                             model_params_flux,
                             model_params_rcur,
                             model_params_vsig,
                             model_params.at("vsys"),
                             model_data);
}

} // namespace model_thindisk
} // namespace gbkfit
