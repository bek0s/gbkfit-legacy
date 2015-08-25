
#include "gbkfit/spread_function.hpp"
#include "gbkfit/math.hpp"

#include <algorithm>

namespace gbkfit {

line_spread_function::line_spread_function(void)
{}

line_spread_function::~line_spread_function()
{}

line_spread_function_gaussian::line_spread_function_gaussian(float fwhm)
    : m_fwhm(fwhm)
{}

line_spread_function_gaussian::~line_spread_function_gaussian()
{}

line_spread_function_array::line_spread_function_array(float* data, std::size_t length, float step)
    : m_data(data,data+length),
      m_length(length),
      m_step(step)
{}

line_spread_function_array::~line_spread_function_array()
{}

void line_spread_function_array::as_array(std::size_t length, float step, float* data) const
{
    if(m_length != length) {
        // throw exception
    }
    if(m_step != step) {
        // throw exception
    }
    std::copy(m_data.begin(),m_data.end(),data);
}

void line_spread_function_gaussian::as_array(std::size_t length, float step, float* data) const
{
    float sigma = math::gaussian_sigma_from_fwhm(m_fwhm);

    for(std::size_t z = 0; z < length; ++z)
    {
        float zn = (static_cast<int>(z)-static_cast<int>(length)/2.0f+0.5f) * step;
        data[z] = math::gaussian_function_1d(zn,1.0f,0.0f,sigma);
    }

    float sum = std::accumulate(data,data+length,0.0f);
    std::for_each(data,data+length,[&sum](float &val)->void{val/=sum;});
}

line_spread_function_moffat::line_spread_function_moffat(float fwhm, float beta)
    : m_fwhm(fwhm),
      m_beta(beta)
{}

line_spread_function_moffat::~line_spread_function_moffat()
{}

void line_spread_function_moffat::as_array(std::size_t length, float step, float* data) const
{
    float alpha = math::moffat_alpha_from_beta_and_fwhm(m_beta,m_fwhm);

    for(std::size_t z = 0; z < length; ++z)
    {
        float zn = (static_cast<int>(z)-static_cast<int>(length)/2.0f+0.5f) * step;
        data[z] = math::moffat_function_1d(zn,1.0f,0.0f,alpha,m_beta);
    }

    float sum = std::accumulate(data,data+length,0.0f);
    std::for_each(data,data+length,[&sum](float &val)->void{val/=sum;});
}

line_spread_function_lorentzian::line_spread_function_lorentzian(float fwhm)
    : m_fwhm(fwhm)
{}

line_spread_function_lorentzian::~line_spread_function_lorentzian()
{}

void line_spread_function_lorentzian::as_array(std::size_t length, float step, float* data) const
{
    float gamma = math::lorentzian_gamma_from_fwhm(m_fwhm);

    for(std::size_t z = 0; z < length; ++z)
    {
        float zn = (static_cast<int>(z)-static_cast<int>(length)/2.0f+0.5f) * step;
        data[z] = math::lorentzian_function_1d(zn,1.0f,0.0f,gamma);
    }

    float sum = std::accumulate(data,data+length,0.0f);
    std::for_each(data,data+length,[&sum](float &val)->void{val/=sum;});
}

point_spread_function::point_spread_function(void)
{}

point_spread_function::~point_spread_function()
{}

point_spread_function_image::point_spread_function_image(float* data, std::size_t length_x, std::size_t length_y, float step_x, float step_y)
    : m_data(data,data+length_x*length_y),
      m_length_x(length_x),
      m_length_y(length_y),
      m_step_x(step_x),
      m_step_y(step_y)
{}

point_spread_function_image::~point_spread_function_image()
{}

void point_spread_function_image::as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const
{
    if(m_length_x != length_x || m_length_y != length_y) {
        // throw exception
    }
    if(m_step_x != step_x || m_step_y != step_y) {
        // throw exception
    }
    std::copy(m_data.begin(),m_data.end(),data);
}

point_spread_function_gaussian::point_spread_function_gaussian(float fwhm)
    : point_spread_function_gaussian(fwhm,fwhm,0.0f)
{}

point_spread_function_gaussian::point_spread_function_gaussian(float fwhm_x, float fwhm_y, float pa)
    : m_fwhm_x(fwhm_x),
      m_fwhm_y(fwhm_y),
      m_pa(pa)
{}

point_spread_function_gaussian::~point_spread_function_gaussian()
{}

void point_spread_function_gaussian::as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const
{
    float sigma_x = math::gaussian_sigma_from_fwhm(m_fwhm_x);
    float sigma_y = math::gaussian_sigma_from_fwhm(m_fwhm_y);

    for(std::size_t y = 0; y < length_y; ++y)
    {
        float yn = (static_cast<int>(y)-static_cast<int>(length_y)/2.0f+0.5f) * step_y;
        for(std::size_t x = 0; x < length_x; ++x)
        {
            float xn = (static_cast<int>(x)-static_cast<int>(length_x)/2.0f+0.5f) * step_x;
            data[y*length_x+x] = math::gaussian_function_2d(xn,yn,1.0f,0.0f,0.0f,sigma_x,sigma_y,math::deg_to_rad(m_pa));
        }
    }

    float sum = std::accumulate(data,data+length_x*length_y,0.0f);
    std::for_each(data,data+length_x*length_y,[&sum](float &val)->void{val/=sum;});
}

point_spread_function_lorentzian::point_spread_function_lorentzian(float fwhm)
    : point_spread_function_lorentzian(fwhm,fwhm,0.0f)
{}

point_spread_function_lorentzian::point_spread_function_lorentzian(float fwhm_x, float fwhm_y, float pa)
    : m_fwhm_x(fwhm_x),
      m_fwhm_y(fwhm_y),
      m_pa(pa)
{}

point_spread_function_lorentzian::~point_spread_function_lorentzian()
{}

void point_spread_function_lorentzian::as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const
{
    float gamma_x = math::lorentzian_gamma_from_fwhm(m_fwhm_x);
    float gamma_y = math::lorentzian_gamma_from_fwhm(m_fwhm_y);

    for(std::size_t y = 0; y < length_y; ++y)
    {
        float yn = (static_cast<int>(y)-static_cast<int>(length_y)/2.0f+0.5f) * step_y;
        for(std::size_t x = 0; x < length_x; ++x)
        {
            float xn = (static_cast<int>(x)-static_cast<int>(length_x)/2.0f+0.5f) * step_x;
            data[y*length_x+x] = math::lorentzian_function_2d(xn,yn,1.0f,0.0f,0.0f,gamma_x,gamma_y,math::deg_to_rad(m_pa));
        }
    }

    float sum = std::accumulate(data,data+length_x*length_y,0.0f);
    std::for_each(data,data+length_x*length_y,[&sum](float &val)->void{val/=sum;});
}

point_spread_function_moffat::point_spread_function_moffat(float fwhm, float beta)
    : point_spread_function_moffat(fwhm,fwhm,0.0f,beta)
{}

point_spread_function_moffat::point_spread_function_moffat(float fwhm_x, float fwhm_y, float pa, float beta)
    : m_fwhm_x(fwhm_x),
      m_fwhm_y(fwhm_y),
      m_beta(beta),
      m_pa(pa)
{}

point_spread_function_moffat::~point_spread_function_moffat()
{}

void point_spread_function_moffat::as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const
{
    float alpha_x = math::moffat_alpha_from_beta_and_fwhm(m_beta,m_fwhm_x);
    float alpha_y = math::moffat_alpha_from_beta_and_fwhm(m_beta,m_fwhm_y);

    for(std::size_t y = 0; y < length_y; ++y)
    {
        float yn = (static_cast<int>(y)-static_cast<int>(length_y)/2.0f+0.5f) * step_y;
        for(std::size_t x = 0; x < length_x; ++x)
        {
            float xn = (static_cast<int>(x)-static_cast<int>(length_x)/2.0f+0.5f) * step_x;
            data[y*length_x+x] = math::moffat_function_2d(xn,yn,1.0f,0.0f,0.0f,alpha_x,alpha_y,m_beta,math::deg_to_rad(m_pa));
        }
    }

    float sum = std::accumulate(data,data+length_x*length_y,0.0f);
    std::for_each(data,data+length_x*length_y,[&sum](float &val)->void{val/=sum;});
}

} // namespace gbkfit
