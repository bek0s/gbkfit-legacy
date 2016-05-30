
#include "gbkfit/fitter_result.hpp"
#include "gbkfit/dmodel.hpp"
#include "gbkfit/gmodel.hpp"
#include "gbkfit/model.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/fits.hpp"
#include <fstream>
#include "jsoncpp/json/json.h"

namespace gbkfit {

FitterResultMode::FitterResultMode(float chisqr,
                                   float reduced_chisqr,
                                   const std::vector<float>& param_values,
                                   const std::vector<float>& param_errors,
                                   std::vector<NDArrayHost*>& models,
                                   std::vector<NDArrayHost*>& residuals)
    : m_chisqr(chisqr)
    , m_reduced_chisqr(reduced_chisqr)
    , m_param_values(param_values)
    , m_param_errors(param_errors)
    , m_models(models)
    , m_residuals(residuals)
{
}

FitterResultMode::~FitterResultMode()
{
    for(auto& model : m_models)
        delete model;
    for(auto& residual : m_residuals)
        delete residual;
}

float FitterResultMode::get_chisqr(void) const
{
    return m_chisqr;
}

float FitterResultMode::get_reduced_chisqr(void) const
{
    return m_reduced_chisqr;
}

const std::vector<float>& FitterResultMode::get_param_values(void) const
{
    return m_param_values;
}

const std::vector<float>& FitterResultMode::get_param_errors(void) const
{
    return m_param_errors;
}

const std::vector<NDArrayHost*>& FitterResultMode::get_models(void) const
{
    return m_models;
}

const std::vector<NDArrayHost*>& FitterResultMode::get_residuals(void) const
{
    return m_residuals;
}

FitterResult::FitterResult(const DModel *dmodel,
                           const std::vector<std::string>& dataset_names,
                           const std::vector<NDArray*>& dataset_data,
                           const std::vector<NDArray*>& dataset_errors,
                           const std::vector<NDArray*>& dataset_masks,
                           const std::vector<bool>& param_fixed_flags,
                           const std::vector<std::vector<float>>& param_values,
                           const std::vector<std::vector<float>>& param_errors)
    : m_dmodel(dmodel)
    , m_fev(0)
    , m_dof(0)
    , m_dataset_names(dataset_names)
    , m_param_names(dmodel->get_galaxy_model()->get_param_names())
    , m_param_fixed_flags(param_fixed_flags)
{
    // Global information
    std::size_t dataset_count = dataset_names.size();
    std::size_t param_count = m_param_names.size();
    std::size_t mode_count = param_values.size();
    NDShape shape =  dataset_data[0]->get_shape();
    std::size_t pixel_count = (std::size_t)shape.get_dim_length_product();

    // Iterate over datasets
    for(std::size_t i = 0; i < dataset_count; ++i)
    {
        // Allocate internal dataset memory
        NDArrayHost* data = new NDArrayHost(shape);
        NDArrayHost* error = new NDArrayHost(shape);
        NDArrayHost* mask = new NDArrayHost(shape);
        m_dataset_data.push_back(data);
        m_dataset_errors.push_back(error);
        m_dataset_masks.push_back(mask);

        // Copy input datasets to internal datasets
        data->write_data(dataset_data[i]);
        error->write_data(dataset_errors[i]);
        mask->write_data(dataset_masks[i]);

        // Calculate degrees of freedom
        float* mask_ptr = m_dataset_masks[i]->get_host_ptr();
        m_dof += std::count(mask_ptr, mask_ptr + pixel_count, 1.0f);
    }

    // Iterate over model parameters
    for(std::size_t i = 0; i < param_count; ++i)
    {
        if (!m_param_fixed_flags[i])
            m_param_names_free.push_back(m_param_names[i]);
        else
            m_param_names_fixed.push_back(m_param_names[i]);
    }

    // Iterate over modes
    for(std::size_t i = 0; i < mode_count; ++i)
    {
        float chisqr = 0;
        float reduced_chisqr = 0;
        std::vector<NDArrayHost*> models;
        std::vector<NDArrayHost*> residuals;

        // Create parameter map
        std::map<std::string, float> param_map;
        for(std::size_t j = 0; j < param_count; ++j)
            param_map[m_param_names[j]] = param_values[i][j];

        // Create model dataset
        std::map<std::string, NDArrayHost*> model_dataset = dmodel->evaluate(param_map);

        // Create residuals and statistics
        for(std::size_t j = 0; j < dataset_count; ++j)
        {
            std::string name = dataset_names[j];

            // Allocate model and residual memory
            NDArrayHost* model = new NDArrayHost(shape);
            NDArrayHost* residual = new NDArrayHost(shape);
            models.push_back(model);
            residuals.push_back(residual);

            // Create a copy of the model data
            model->write_data(model_dataset[name]);

            // Create convenience pointers
            float* data_ptr = m_dataset_data[j]->get_host_ptr();
            float* error_ptr = m_dataset_errors[j]->get_host_ptr();
            float* mask_ptr = m_dataset_masks[j]->get_host_ptr();
            float* model_ptr = model->get_host_ptr();
            float* residual_ptr = residual->get_host_ptr();

            // Calculate residual and statistics
            for(std::size_t k = 0; k < pixel_count; ++k)
            {
                if (mask_ptr[k] > 0)
                {
                    residual_ptr[k] = data_ptr[k] - model_ptr[k];
                    chisqr += (residual_ptr[k]*residual_ptr[k])/(error_ptr[k]*error_ptr[k]);
                }
                else
                {
                    residual_ptr[k] = std::numeric_limits<float>::quiet_NaN();
                }
            }
            reduced_chisqr = chisqr / m_dof;
        }

        //  Create and add the new mode
        m_modes.push_back(new FitterResultMode(chisqr,
                                               reduced_chisqr,
                                               param_values[i],
                                               param_errors[i],
                                               models,
                                               residuals));
    }
}

FitterResult::~FitterResult()
{
    for(auto& data : m_dataset_data)
        delete data;
    for(auto& error : m_dataset_errors)
        delete error;
    for(auto& mask : m_dataset_masks)
        delete mask;
    for(auto& mode : m_modes)
        delete mode;
}

int FitterResult::get_fev(void) const
{
    return m_fev;
}

int FitterResult::get_dof(void) const
{
    return m_dof;
}

std::size_t FitterResult::get_dataset_count(void) const
{
    return m_dataset_names.size();
}

const std::vector<std::string>& FitterResult::get_dataset_names(void) const
{
    return m_dataset_names;
}

const std::vector<NDArrayHost*>& FitterResult::get_dataset_data(void) const
{
    return m_dataset_data;
}

const std::vector<NDArrayHost*>& FitterResult::get_dataset_errors(void) const
{
    return m_dataset_errors;
}

const std::vector<NDArrayHost*>& FitterResult::get_dataset_masks(void) const
{
    return m_dataset_masks;
}

std::size_t FitterResult::get_param_count(void) const
{
    return m_param_names.size();
}

std::size_t FitterResult::get_param_count_fixed(void) const
{
    return m_param_names_fixed.size();
}

std::size_t FitterResult::get_param_count_free(void) const
{
    return m_param_names_free.size();
}

const std::vector<std::string>& FitterResult::get_param_names(void) const
{
    return m_param_names;
}

const std::vector<std::string>& FitterResult::get_param_names_free(void) const
{
    return m_param_names_free;
}

const std::vector<std::string>& FitterResult::get_param_names_fixed(void) const
{
    return m_param_names_fixed;
}

std::size_t FitterResult::get_mode_count(void) const
{
    return m_modes.size();
}

const FitterResultMode& FitterResult::get_mode(std::size_t index) const
{
    return *m_modes.at(index);
}

std::string FitterResult::to_string(void) const
{
    std::size_t mode_count = get_mode_count();
    std::size_t param_count_all = get_param_count();
    std::size_t param_count_free = get_param_count_free();
    std::size_t param_count_fixed = get_param_count_fixed();

    std::ostringstream str;
    str << "dof: " << m_dof << std::endl;
    str << "number of modes: " << mode_count << std::endl;
    str << "number of parameters (all): " <<  param_count_all << std::endl;
    str << "number of parameters (free): " <<  param_count_free << std::endl;
    str << "number of parameters (fixed): " <<  param_count_fixed << std::endl;

    for(std::size_t i = 0; i < mode_count; ++i)
    {
        const FitterResultMode& mode = get_mode(i);
        std::vector<float> param_values = mode.get_param_values();
        std::vector<float> param_errors = mode.get_param_errors();
        float chisqr = mode.get_chisqr();
        float reduced_chisqr = mode.get_reduced_chisqr();

        str << "mode " << i << ": " << std::endl;

        str << std::fixed
            << std::setprecision(2);

        str << "chi-squared: " << chisqr << std::endl
            << "chi-squared (reduced): " << reduced_chisqr << std::endl;

        for(std::size_t j = 0; j < param_count_all; ++j)
        {
            str << "name: "  << std::setw(4) << m_param_names[j]       << ", "
                << "fixed: " << std::setw(1) << m_param_fixed_flags[j] << ", "
                << "best: "  << std::setw(8) << param_values[j]        << ", "
                << "error: " << std::setw(8) << param_errors[j]        << std::endl;
        }
    }

    return str.str();
}

void FitterResult::save(const std::string& filename) const
{
    std::size_t dataset_count = get_dataset_count();
    std::size_t param_count = get_param_count();
    std::size_t mode_count = get_mode_count();

    Json::Value root;

    //
    // Global results
    //

    root["model"] = m_dmodel->get_type();
    root["dof"] = m_dof;

    //
    // Datasets
    //
    Json::Value datasets;
    for(std::size_t i = 0; i < dataset_count; ++i)
    {
        datasets.append(m_dataset_names[i]);

        std::string dataset_name = m_dataset_names[i];
        std::ostringstream file_dat;
        std::ostringstream file_err;
        std::ostringstream file_msk;

        file_dat << "!" << dataset_name << "_data.fits";
        file_err << "!" << dataset_name << "_data_error.fits";
        file_msk << "!" << dataset_name << "_data_mask.fits";

        fits::write_to(file_dat.str(), *m_dataset_data[i]);
        fits::write_to(file_err.str(), *m_dataset_errors[i]);
        fits::write_to(file_msk.str(), *m_dataset_masks[i]);
    }
    root["datasets"] = datasets;

    //
    // Parameters
    //
    Json::Value parameters;
    for(std::size_t i = 0; i < param_count; ++i) {
        Json::Value parameter;
        parameter["name"] = m_param_names[i];
        parameter["fixed"] = m_param_fixed_flags[i];
        parameters.append(parameter);
    }
    root["parameters"] = parameters;

    //
    // Modes
    //
    Json::Value json_modes;
    for(std::size_t i = 0; i < mode_count; ++i)
    {
        const FitterResultMode& mode = get_mode(i);
        Json::Value json_mode;
        Json::Value json_mode_parameters;

        json_mode["chisqr"] = mode.get_chisqr();
        json_mode["rchisqr"] = mode.get_reduced_chisqr();

        for(std::size_t j = 0; j < param_count; ++j)
        {
            Json::Value json_mode_parameter;
            json_mode_parameter["name"] = m_param_names[j];
            json_mode_parameter["value"] = mode.get_param_values()[j];
            json_mode_parameter["error"] = mode.get_param_errors()[j];
            json_mode_parameters.append(json_mode_parameter);
        }

        json_mode["parameters"] = json_mode_parameters;
        json_modes.append(json_mode);

        for(std::size_t j = 0; j < dataset_count; ++j)
        {
            std::string dataset_name = m_dataset_names[j];
            std::ostringstream file_mdl;
            std::ostringstream file_res;
            file_mdl << "!mode_" << i << "_" << dataset_name << "_model.fits";
            file_res << "!mode_" << i << "_" << dataset_name << "_residual.fits";
            fits::write_to(file_mdl.str(), *(get_mode(i).get_models()[j]));
            fits::write_to(file_res.str(), *(get_mode(i).get_residuals()[j]));
        }

    }
    root["modes"] = json_modes;


    Json::StyledWriter writer;
    std::ofstream ofstream(filename.c_str(), std::ios::out);
    ofstream << writer.write(root);
    ofstream.close();
}

} // namespace gbkfit
