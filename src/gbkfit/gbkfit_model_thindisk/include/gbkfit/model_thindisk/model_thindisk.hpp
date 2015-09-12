#pragma once
#ifndef GBKFIT_MODEL_THINDISK_HPP
#define GBKFIT_MODEL_THINDISK_HPP

#include "gbkfit/model.hpp"
#include "gbkfit/ndarray.hpp"

#include <complex>
#include <map>
#include <string>

#include <fftw3.h>
#include <valarray>

namespace gbkfit {

class model_param_fit_info
{
    float value;
    float value_init;
    float value_min;
    float value_max;
};

//!
//! \brief The model class
//!
class model3
{

public:

    model3(void) {}

    virtual ~model3() {}

    virtual const std::string& get_type_name(void) const = 0;

    virtual std::vector<std::string> get_parameter_names(void) const = 0;

    virtual std::size_t get_model_data_length(void) const = 0;

    virtual void evaluate(const std::map<std::string,float>& model_params, std::valarray<float>& model_data) = 0;

};

namespace model_thindisk {
// todo: check the aligned velmap
class model_thindisk : public gbkfit::model3
{

protected:

    std::size_t m_intcube_width;
    std::size_t m_intcube_height;
    std::size_t m_intcube_depth;

    float m_step_x;
    float m_step_y;
    float m_step_z;

    std::size_t m_upsampling_x;
    std::size_t m_upsampling_y;
    std::size_t m_upsampling_z;

    std::size_t m_intcube_aligned_width;
    std::size_t m_intcube_aligned_height;
    std::size_t m_intcube_aligned_depth;

    std::size_t m_psf_width;
    std::size_t m_psf_height;
    std::size_t m_psf_depth;

    float* m_h_intcube;
    float* m_h_intcube_aligned;
    std::complex<float>* m_h_intcube_aligned_fftw3;

    float* m_h_psf;
    float* m_h_psf_aligned;
    std::complex<float>* m_h_psf_aligned_fftw3;

    float* m_h_velmap;
    float* m_h_sigmap;

    float* m_h_velmap_aligned;
    float* m_h_sigmap_aligned;

    fftwf_plan m_fft_plan_psf_r2c;
    fftwf_plan m_fft_plan_intcube_r2c;
    fftwf_plan m_fft_plan_intcube_c2r;

public:

    model_thindisk(std::size_t width,
                   std::size_t height,
                   std::size_t depth,
                   float step_x,
                   float step_y,
                   float step_z,
                   std::size_t upsampling_x,
                   std::size_t upsampling_y,
                   std::size_t upsampling_z,
                   const gbkfit::ndarray* psf);

    ~model_thindisk();

    virtual const std::string& get_type_name(void) const = 0;

    virtual std::vector<std::string> get_parameter_names(void) const = 0;

    std::size_t get_model_data_length(void) const final;

    virtual void evaluate(const std::map<std::string,float>& model_params, std::valarray<float>& model_data) = 0;

    void evaluate(unsigned int model_id,
                  const std::vector<float>& model_params_proj,
                  const std::vector<float>& model_params_flux,
                  const std::vector<float>& model_params_rcur,
                  const std::vector<float>& model_params_vsig,
                  float vsys,
                  std::valarray<float>& model_data);

private:

    void get_intcube_margins(int& width_0, int& width_1, int& height_0, int& height_1, int& depth_0, int& depth_1) const;

public:

    void get_intcube_aligned_lengths(std::size_t& width, std::size_t& height, std::size_t& depth) const;

};

} // namespace model_thindisk
} // namespace gbkfit

#endif // GBKFIT_MODEL_THINDISK_HPP
