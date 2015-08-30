#pragma once
#ifndef GBKFIT_MODEL_THINDISK_EPINAT_HPP
#define GBKFIT_MODEL_THINDISK_EPINAT_HPP

#include "gbkfit/model_thindisk/model_thindisk.hpp"

namespace gbkfit {
namespace model_thindisk {

class model_thindisk_epinat : public gbkfit::model_thindisk::model_thindisk
{

protected:


public:

    model_thindisk_epinat(std::size_t width,
                          std::size_t height,
                          std::size_t depth,
                          float step_x,
                          float step_y,
                          float step_z,
                          std::size_t upsampling_x,
                          std::size_t upsampling_y,
                          std::size_t upsampling_z,
                          const gbkfit::ndarray* psf);

    ~model_thindisk_epinat();

    const std::string& get_type_name(void) const final;

    std::vector<std::string> get_parameter_names(void) const final;

    void evaluate(const std::map<std::string,float>& model_params, std::valarray<float>& model_data) final;

};

} // namespace model_thindisk
} // namespace gbkfit

#endif // GBKFIT_MODEL_THINDISK_EPINAT_HPP
