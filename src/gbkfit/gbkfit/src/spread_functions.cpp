
#include "gbkfit/spread_functions.hpp"
#include "gbkfit/math.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/utility.hpp"

namespace gbkfit {

template <typename T>
void evaluate_data_1d(float* data, int size, float step, T func)
{
    for(int z = 0; z < size; ++z)
    {
        float zn = (z - size / 2.0f + 0.5f) * step;

        data[z] = func(zn);
    }

    math::normalize_integral(data, size);
}

template <typename T>
void evaluate_data_2d(float* data, int size_x, int size_y, float step_x, float step_y, T func)
{
    for(int y = 0; y < size_y; ++y)
    {
        float yn = (y - size_y / 2.0f + 0.5f) * step_y;

        for(int x = 0; x < size_x; ++x)
        {
            float xn = (x - size_x / 2.0f + 0.5f) * step_x;

            data[y*size_x+x] = func(xn, yn);
        }
    }

    math::normalize_integral(data, size_x*size_y);
}

line_spread_function::line_spread_function(void)
{
}

line_spread_function::~line_spread_function()
{
}

ndarray_host* line_spread_function::as_array(float step) const
{
    ndshape size = get_recommended_size(step);
    return as_array(size[0],step);
}

ndarray_host* line_spread_function::as_array(int size, float step) const
{
    ndarray_host* data = new ndarray_host_new({size});
    as_array(size, step, data->get_host_ptr());
    return data;
}

line_spread_function_array::line_spread_function_array(float* data, float step, int size)
    : m_data(nullptr)
    , m_step(step)
    , m_size(size)
{
    m_data = new float[m_size];
    std::copy(data, data+size, m_data);
}

line_spread_function_array::~line_spread_function_array()
{
    delete [] m_data;
}

ndshape line_spread_function_array::get_recommended_size(float step) const
{
    if (step != m_step) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return ndshape({m_size});
}

void line_spread_function_array::as_array(int size, float step, float* data) const
{
    if (size != m_size) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    if (step != m_step) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    std::copy_n(m_data, m_size, data);
}

line_spread_function_gaussian::line_spread_function_gaussian(float fwhm)
    : m_fwhm(fwhm)
{
}

line_spread_function_gaussian::~line_spread_function_gaussian()
{
}

void line_spread_function_gaussian::as_array(int size, float step, float* data) const
{
    float sigma = math::gaussian_sigma_from_fwhm(m_fwhm);

    auto eval_func = std::bind(math::gaussian_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               sigma);

    evaluate_data_1d(data, size, step, eval_func);
}

ndshape line_spread_function_gaussian::get_recommended_size(float step) const
{
    return ndshape({util_num::roundu_odd(5*m_fwhm/step)});
}

line_spread_function_lorentzian::line_spread_function_lorentzian(float fwhm)
    : m_fwhm(fwhm)
{
}

line_spread_function_lorentzian::~line_spread_function_lorentzian()
{
}

ndshape line_spread_function_lorentzian::get_recommended_size(float step) const
{
    return ndshape({util_num::roundu_odd(5*m_fwhm/step)});
}

void line_spread_function_lorentzian::as_array(int size, float step, float* data) const
{
    float gamma = math::lorentzian_gamma_from_fwhm(m_fwhm);

    auto eval_func = std::bind(math::lorentzian_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               gamma);

    evaluate_data_1d(data, size, step, eval_func);
}

line_spread_function_moffat::line_spread_function_moffat(float fwhm, float beta)
    : m_fwhm(fwhm)
    , m_beta(beta)
{
}

line_spread_function_moffat::~line_spread_function_moffat()
{
}

ndshape line_spread_function_moffat::get_recommended_size(float step) const
{
    return ndshape({util_num::roundu_odd(5*m_fwhm/step)});
}

void line_spread_function_moffat::as_array(int size, float step, float *data) const
{
    float alpha = math::moffat_alpha_from_beta_and_fwhm(m_beta, m_fwhm);

    auto eval_func = std::bind(math::moffat_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               alpha,
                               m_beta);

    evaluate_data_1d(data, size, step, eval_func);
}

point_spread_function::point_spread_function(void)
{
}

point_spread_function::~point_spread_function()
{
}

ndarray_host* point_spread_function::as_image(float step_x, float step_y) const
{
    ndshape shape = get_recommended_size(step_x, step_y);
    return as_image(shape[0], shape[1], step_x, step_y);
}

ndarray_host* point_spread_function::as_image(int size_x, int size_y, float step_x, float step_y) const
{
    ndarray_host* data = new ndarray_host_new({size_x, size_y});
    as_image(size_x, size_y, step_x, step_y, data->get_host_ptr());
    return data;
}

point_spread_function_image::point_spread_function_image(float* data, int size_x, int size_y, float step_x, float step_y)
    : m_data(nullptr)
    , m_step_x(step_x)
    , m_step_y(step_y)
    , m_size_x(size_x)
    , m_size_y(size_y)
{
    m_data = new float[m_size_x*m_size_y];
    std::copy(data, data+size_x*size_y, m_data);
}

point_spread_function_image::~point_spread_function_image()
{
    delete [] m_data;
}

ndshape point_spread_function_image::get_recommended_size(float step_x, float step_y) const
{
    if (step_x != m_step_x || step_y != m_step_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return ndshape({m_size_x, m_size_y});
}

void point_spread_function_image::as_image(int size_x, int size_y, float step_x, float step_y, float* data) const
{
    if (size_x != m_size_x || size_y != m_size_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    if (step_x != m_step_x || step_y != m_step_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    std::copy_n(m_data, m_size_x*m_size_y, data);
}

point_spread_function_gaussian::point_spread_function_gaussian(float fwhm)
    : point_spread_function_gaussian(fwhm, fwhm, 0)
{
}

point_spread_function_gaussian::point_spread_function_gaussian(float fwhm_x, float fwhm_y, float pa)
    : m_fwhm_x(fwhm_x)
    , m_fwhm_y(fwhm_y)
    , m_pa(pa)
{
}

point_spread_function_gaussian::~point_spread_function_gaussian()
{
}

ndshape point_spread_function_gaussian::get_recommended_size(float step_x, float step_y) const
{
    return ndshape({util_num::roundu_odd(5*m_fwhm_x/step_x),
                    util_num::roundu_odd(5*m_fwhm_y/step_y)});
}

void point_spread_function_gaussian::as_image(int size_x, int size_y, float step_x, float step_y, float* data) const
{
    float sigma_x = math::gaussian_sigma_from_fwhm(m_fwhm_x);
    float sigma_y = math::gaussian_sigma_from_fwhm(m_fwhm_y);

    auto eval_func = std::bind(math::gaussian_function_2d<float>,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               1.0f,
                               0.0f, 0.0f,
                               sigma_x, sigma_y,
                               math::deg_to_rad(m_pa));

    evaluate_data_2d(data, size_x, size_y, step_x, step_y, eval_func);
}

point_spread_function_lorentzian::point_spread_function_lorentzian(float fwhm)
    : point_spread_function_lorentzian(fwhm, fwhm, 0)
{
}

point_spread_function_lorentzian::point_spread_function_lorentzian(float fwhm_x, float fwhm_y, float pa)
    : m_fwhm_x(fwhm_x)
    , m_fwhm_y(fwhm_y)
    , m_pa(pa)
{
}

point_spread_function_lorentzian::~point_spread_function_lorentzian()
{
}

ndshape point_spread_function_lorentzian::get_recommended_size(float step_x, float step_y) const
{
    return ndshape({util_num::roundu_odd(5*m_fwhm_x/step_x),
                    util_num::roundu_odd(5*m_fwhm_y/step_y)});
}

void point_spread_function_lorentzian::as_image(int size_x, int size_y, float step_x, float step_y, float* data) const
{
    float gamma_x = math::lorentzian_gamma_from_fwhm(m_fwhm_x);
    float gamma_y = math::lorentzian_gamma_from_fwhm(m_fwhm_y);

    auto eval_func = std::bind(math::lorentzian_function_2d<float>,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               1.0f,
                               0.0f, 0.0f,
                               gamma_x, gamma_y,
                               math::deg_to_rad(m_pa));

    evaluate_data_2d(data, size_x, size_y, step_x, step_y, eval_func);
}

point_spread_function_moffat::point_spread_function_moffat(float fwhm, float beta)
    : point_spread_function_moffat(fwhm, fwhm, 0, beta)
{
}

point_spread_function_moffat::point_spread_function_moffat(float fwhm_x, float fwhm_y, float pa, float beta)
    : m_fwhm_x(fwhm_x)
    , m_fwhm_y(fwhm_y)
    , m_beta(beta)
    , m_pa(pa)
{
}

point_spread_function_moffat::~point_spread_function_moffat()
{
}

ndshape point_spread_function_moffat::get_recommended_size(float step_x, float step_y) const
{
    return ndshape({util_num::roundu_odd(5*m_fwhm_x/step_x),
                    util_num::roundu_odd(5*m_fwhm_y/step_y)});
}

void point_spread_function_moffat::as_image(int size_x, int size_y, float step_x, float step_y, float* data) const
{
    float alpha_x = math::moffat_alpha_from_beta_and_fwhm(m_beta, m_fwhm_x);
    float alpha_y = math::moffat_alpha_from_beta_and_fwhm(m_beta, m_fwhm_y);

    auto eval_func = std::bind(math::moffat_function_2d<float>,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               1.0f,
                               0.0f, 0.0f,
                               alpha_x, alpha_y,
                               m_beta,
                               math::deg_to_rad(m_pa));

    evaluate_data_2d(data, size_x, size_y, step_x, step_y, eval_func);
}

} // namespace gbkfit
