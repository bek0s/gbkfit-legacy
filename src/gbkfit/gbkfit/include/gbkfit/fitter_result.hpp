#pragma once
#ifndef GBKFIT_FITTER_RESULT_HPP
#define GBKFIT_FITTER_RESULT_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class FitterResultMode
{

protected:

    float m_chisqr;
    float m_reduced_chisqr;
    std::vector<float> m_param_values;
    std::vector<float> m_param_errors;
    std::vector<NDArrayHost*> m_models;
    std::vector<NDArrayHost*> m_residuals;

public:

    FitterResultMode(float chisqr,
                     float reduced_chisqr,
                     const std::vector<float>& param_values,
                     const std::vector<float>& param_errors,
                     std::vector<NDArrayHost*>& models,
                     std::vector<NDArrayHost*>& residuals);

    virtual ~FitterResultMode();

    float get_chisqr(void) const;

    float get_reduced_chisqr(void) const;

    const std::vector<float>& get_param_values(void) const;

    const std::vector<float>& get_param_errors(void) const;

    const std::vector<NDArrayHost*>& get_models(void) const;

    const std::vector<NDArrayHost*>& get_residuals(void) const;

}; // class FitterResultMode

class FitterResult
{

private:

    const DModel* m_dmodel;

    int m_fev;
    int m_dof;

    std::vector<std::string> m_dataset_names;
    std::vector<NDArrayHost*> m_dataset_data;
    std::vector<NDArrayHost*> m_dataset_errors;
    std::vector<NDArrayHost*> m_dataset_masks;

    std::vector<std::string> m_param_names;
    std::vector<std::string> m_param_names_free;
    std::vector<std::string> m_param_names_fixed;
    std::vector<bool> m_param_fixed_flags;

    std::vector<FitterResultMode*> m_modes;

public:

    FitterResult(const DModel* dmodel,
                 const std::vector<std::string>& dataset_names,
                 const std::vector<NDArray*>& dataset_data,
                 const std::vector<NDArray*>& dataset_errors,
                 const std::vector<NDArray*>& dataset_masks,
                 const std::vector<bool>& param_fixed_flags,
                 const std::vector<std::vector<float>>& param_values,
                 const std::vector<std::vector<float>>& param_error);

    virtual ~FitterResult();

    int get_fev(void) const;

    int get_dof(void) const;

    std::size_t get_dataset_count(void) const;

    const std::vector<std::string>& get_dataset_names(void) const;

    const std::vector<NDArrayHost*>& get_dataset_data(void) const;

    const std::vector<NDArrayHost*>& get_dataset_errors(void) const;

    const std::vector<NDArrayHost*>& get_dataset_masks(void) const;

    std::size_t get_param_count(void) const;

    std::size_t get_param_count_fixed(void) const;

    std::size_t get_param_count_free(void) const;

    const std::vector<std::string>& get_param_names(void) const;

    const std::vector<std::string>& get_param_names_free(void) const;

    const std::vector<std::string>& get_param_names_fixed(void) const;

    std::size_t get_mode_count(void) const;

    const FitterResultMode& get_mode(std::size_t index) const;

    std::string to_string(void) const;

    void save(const std::string& filename, const std::string& output_dir) const;

}; // class FitterResult

} // namespace gbkfit

#endif // GBKFIT_FITTER_RESULT_HPP
