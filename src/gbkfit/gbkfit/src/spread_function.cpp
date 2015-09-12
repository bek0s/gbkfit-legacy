
#include "gbkfit/spread_function.hpp"
#include "gbkfit/math.hpp"
#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {

template <typename T>
void evaluate_data_1d(float* data, float step, int size, T func)
{
    for(int z = 0; z < size; ++z)
    {
        float zn = (z - size / 2.0f + 0.5f) * step;

        data[z] = func(zn);
    }

    math::normalize_integral(data, size);
}

template <typename T>
void evaluate_data_2d(float* data, float step_x, float step_y, int size_x, int size_y, T func)
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

line_spread_function_gaussian::line_spread_function_gaussian(float fwhm)
    : m_fwhm(fwhm)
{
}

line_spread_function_gaussian::~line_spread_function_gaussian()
{
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

ndarray_host* line_spread_function_array::as_array(float step) const
{
    return as_array(step, m_size);
}

ndarray_host* line_spread_function_array::as_array(float step, int size) const
{
    if (size != m_size) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    if (step != m_step) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return new ndarray_host_new({m_size}, m_data);
}

ndarray_host* line_spread_function_gaussian::as_array(float step) const
{
    return as_array(step, 17);
}

ndarray_host* line_spread_function_gaussian::as_array(float step, int size) const
{
    float sigma = math::gaussian_sigma_from_fwhm(m_fwhm);

    ndarray_host* data = new ndarray_host_new({size});

    auto eval_func = std::bind(math::gaussian_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               sigma);

    evaluate_data_1d(data->get_host_ptr(), step, size, eval_func);

    return data;
}

line_spread_function_lorentzian::line_spread_function_lorentzian(float fwhm)
    : m_fwhm(fwhm)
{
}

line_spread_function_lorentzian::~line_spread_function_lorentzian()
{
}

ndarray_host* line_spread_function_lorentzian::as_array(float step) const
{
    return as_array(step, 17);
}

ndarray_host* line_spread_function_lorentzian::as_array(float step, int size) const
{
    float gamma = math::lorentzian_gamma_from_fwhm(m_fwhm);

    ndarray_host* data = new ndarray_host_new({size});

    auto eval_func = std::bind(math::lorentzian_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               gamma);

    evaluate_data_1d(data->get_host_ptr(), step, size, eval_func);

    return data;
}

line_spread_function_moffat::line_spread_function_moffat(float fwhm, float beta)
    : m_fwhm(fwhm)
    , m_beta(beta)
{
}

line_spread_function_moffat::~line_spread_function_moffat()
{
}

ndarray_host* line_spread_function_moffat::as_array(float step) const
{
    return as_array(step, 17);
}

ndarray_host* line_spread_function_moffat::as_array(float step, int size) const
{
    float alpha = math::moffat_alpha_from_beta_and_fwhm(m_beta, m_fwhm);

    ndarray_host* data = new ndarray_host_new({size});

    auto eval_func = std::bind(math::moffat_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               alpha,
                               m_beta);

    evaluate_data_1d(data->get_host_ptr(), step, size, eval_func);

    return data;
}

point_spread_function::point_spread_function(void)
{
}

point_spread_function::~point_spread_function()
{
}

point_spread_function_image::point_spread_function_image(float* data, float step_x, float step_y, int size_x, int size_y)
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

ndarray_host* point_spread_function_image::as_image(float step_x, float step_y) const
{
    return as_image(step_x, step_y, m_size_x, m_size_y);
}

ndarray_host* point_spread_function_image::as_image(float step_x, float step_y, int size_x, int size_y) const
{
    if (size_x != m_size_x || size_y != m_size_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    if (step_x != m_step_x || step_y != m_step_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return new ndarray_host_new({m_size_x, m_size_y}, m_data);
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

ndarray_host* point_spread_function_gaussian::as_image(float step_x, float step_y) const
{
    return as_image(step_x, step_y, 17, 17);
}

ndarray_host* point_spread_function_gaussian::as_image(float step_x, float step_y, int size_x, int size_y) const
{
    float sigma_x = math::gaussian_sigma_from_fwhm(m_fwhm_x);
    float sigma_y = math::gaussian_sigma_from_fwhm(m_fwhm_y);

    ndarray_host* data = new ndarray_host_new({size_x, size_y});

    auto eval_func = std::bind(math::gaussian_function_2d<float>,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               1.0f,
                               0.0f, 0.0f,
                               sigma_x, sigma_y,
                               math::deg_to_rad(m_pa));

    evaluate_data_2d(data->get_host_ptr(), step_x, step_y, size_x, size_y, eval_func);

    return data;
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

ndarray_host* point_spread_function_lorentzian::as_image(float step_x, float step_y) const
{
    return as_image(step_x, step_y, 17, 17);
}

ndarray_host* point_spread_function_lorentzian::as_image(float step_x, float step_y, int size_x, int size_y) const
{
    float gamma_x = math::lorentzian_gamma_from_fwhm(m_fwhm_x);
    float gamma_y = math::lorentzian_gamma_from_fwhm(m_fwhm_y);

    ndarray_host* data = new ndarray_host_new({size_x, size_y});

    auto eval_func = std::bind(math::lorentzian_function_2d<float>,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               1.0f,
                               0.0f, 0.0f,
                               gamma_x, gamma_y,
                               math::deg_to_rad(m_pa));

    evaluate_data_2d(data->get_host_ptr(), step_x, step_y, size_x, size_y, eval_func);

    return data;
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

ndarray_host* point_spread_function_moffat::as_image(float step_x, float step_y) const
{
    return as_image(step_x, step_y, 17, 17);
}

ndarray_host* point_spread_function_moffat::as_image(float step_x, float step_y, int size_x, int size_y) const
{
    float alpha_x = math::moffat_alpha_from_beta_and_fwhm(m_beta, m_fwhm_x);
    float alpha_y = math::moffat_alpha_from_beta_and_fwhm(m_beta, m_fwhm_y);

    ndarray_host* data = new ndarray_host_new({size_x, size_y});

    auto eval_func = std::bind(math::moffat_function_2d<float>,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               1.0f,
                               0.0f, 0.0f,
                               alpha_x, alpha_y,
                               m_beta,
                               math::deg_to_rad(m_pa));

    evaluate_data_2d(data->get_host_ptr(), step_x, step_y, size_x, size_y, eval_func);

    return data;
}

} // namespace gbkfit
