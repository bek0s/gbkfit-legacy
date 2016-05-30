
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

    //util_image::image_flip_2d(data, size_x, size_y);

    math::normalize_integral(data, size_x*size_y);
}

LineSpreadFunction::LineSpreadFunction(void)
{
}

LineSpreadFunction::~LineSpreadFunction()
{
}

std::unique_ptr<NDArrayHost> LineSpreadFunction::as_array(float step) const
{
    NDShape size = get_size(step);
    return as_array(step, size[0]);
}

std::unique_ptr<NDArrayHost> LineSpreadFunction::as_array(float step, int size) const
{
    std::unique_ptr<NDArrayHost> data = std::make_unique<NDArrayHost>(NDShape({size}));
    as_array(step, size, data->get_host_ptr());
    return data;
}

LineSpreadFunctionNone::LineSpreadFunctionNone(void)
{
}

LineSpreadFunctionNone::~LineSpreadFunctionNone()
{
}

NDShape LineSpreadFunctionNone::get_size(float step) const
{
    (void)step;
    return NDShape({1});
}

void LineSpreadFunctionNone::as_array(float step, int size, float* data) const
{
    (void)step;
    std::fill_n(data, size, 0);
    int idx = size/2;
    data[idx] = 1;
}

LineSpreadFunctionArray::LineSpreadFunctionArray(const float* data, float step, int size)
    : m_data(nullptr)
    , m_step(step)
    , m_size(size)
{
    m_data = new float[m_size];
    std::copy(data, data+size, m_data);
}

LineSpreadFunctionArray::~LineSpreadFunctionArray()
{
    delete [] m_data;
}

NDShape LineSpreadFunctionArray::get_size(float step) const
{
    if (step != m_step) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return NDShape({m_size});
}

void LineSpreadFunctionArray::as_array(float step, int size, float* data) const
{
    if (size != m_size) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    if (step != m_step) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    std::copy_n(m_data, m_size, data);
}

LineSpreadFunctionGaussian::LineSpreadFunctionGaussian(float fwhm)
    : m_fwhm(fwhm)
{
}

LineSpreadFunctionGaussian::~LineSpreadFunctionGaussian()
{
}

void LineSpreadFunctionGaussian::as_array(float step, int size, float* data) const
{
    float sigma = math::gaussian_sigma_from_fwhm(m_fwhm);

    auto eval_func = std::bind(math::gaussian_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               sigma);

    evaluate_data_1d(data, size, step, eval_func);
}

NDShape LineSpreadFunctionGaussian::get_size(float step) const
{
    return NDShape({util_num::roundu_odd(5*m_fwhm/step)});
}

LineSpreadFunctionLorentzian::LineSpreadFunctionLorentzian(float fwhm)
    : m_fwhm(fwhm)
{
}

LineSpreadFunctionLorentzian::~LineSpreadFunctionLorentzian()
{
}

NDShape LineSpreadFunctionLorentzian::get_size(float step) const
{
    return NDShape({util_num::roundu_odd(5*m_fwhm/step)});
}

void LineSpreadFunctionLorentzian::as_array(float step, int size, float* data) const
{
    float gamma = math::lorentzian_gamma_from_fwhm(m_fwhm);

    auto eval_func = std::bind(math::lorentzian_function_1d<float>,
                               std::placeholders::_1,
                               1.0f,
                               0.0f,
                               gamma);

    evaluate_data_1d(data, size, step, eval_func);
}

LineSpreadFunctionMoffat::LineSpreadFunctionMoffat(float fwhm, float beta)
    : m_fwhm(fwhm)
    , m_beta(beta)
{
}

LineSpreadFunctionMoffat::~LineSpreadFunctionMoffat()
{
}

NDShape LineSpreadFunctionMoffat::get_size(float step) const
{
    return NDShape({util_num::roundu_odd(5*m_fwhm/step)});
}

void LineSpreadFunctionMoffat::as_array(float step, int size, float *data) const
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

PointSpreadFunction::PointSpreadFunction(void)
{
}

PointSpreadFunction::~PointSpreadFunction()
{
}

std::unique_ptr<NDArrayHost> PointSpreadFunction::as_image(float step_x, float step_y) const
{
    NDShape shape = get_size(step_x, step_y);
    return as_image(step_x, step_y, shape[0], shape[1]);
}

std::unique_ptr<NDArrayHost> PointSpreadFunction::as_image(float step_x, float step_y, int size_x, int size_y) const
{
    std::unique_ptr<NDArrayHost> data = std::make_unique<NDArrayHost>(NDShape({size_x, size_y}));
    as_image(step_x, step_y, size_x, size_y, data->get_host_ptr());
    return data;
}

PointSpreadFunctionNone::PointSpreadFunctionNone(void)
{
}

PointSpreadFunctionNone::~PointSpreadFunctionNone()
{
}

NDShape PointSpreadFunctionNone::get_size(float step_x, float step_y) const
{
    (void)step_x;
    (void)step_y;
    return NDShape({1, 1});
}

void PointSpreadFunctionNone::as_image(float step_x, float step_y, int size_x, int size_y, float* data) const
{
    (void)step_x;
    (void)step_y;
    std::fill_n(data, size_x*size_y, 0);
    int idx = size_y/2*size_x + size_x/2;
    data[idx] = 1;
}

PointSpreadFunctionImage::PointSpreadFunctionImage(const float* data, float step_x, float step_y, int size_x, int size_y)
    : m_data(nullptr)
    , m_step_x(step_x)
    , m_step_y(step_y)
    , m_size_x(size_x)
    , m_size_y(size_y)
{
    m_data = new float[m_size_x*m_size_y];
    std::copy(data, data+size_x*size_y, m_data);
}

PointSpreadFunctionImage::~PointSpreadFunctionImage()
{
    delete [] m_data;
}

NDShape PointSpreadFunctionImage::get_size(float step_x, float step_y) const
{
    if (step_x != m_step_x || step_y != m_step_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return NDShape({m_size_x, m_size_y});
}

void PointSpreadFunctionImage::as_image(float step_x, float step_y, int size_x, int size_y, float* data) const
{
    if (size_x != m_size_x || size_y != m_size_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    if (step_x != m_step_x || step_y != m_step_y) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    std::copy_n(m_data, m_size_x*m_size_y, data);
}

PointSpreadFunctionGaussian::PointSpreadFunctionGaussian(float fwhm)
    : PointSpreadFunctionGaussian(fwhm, fwhm, 0)
{
}

PointSpreadFunctionGaussian::PointSpreadFunctionGaussian(float fwhm_x, float fwhm_y, float pa)
    : m_fwhm_x(fwhm_x)
    , m_fwhm_y(fwhm_y)
    , m_pa(pa)
{
}

PointSpreadFunctionGaussian::~PointSpreadFunctionGaussian()
{
}

NDShape PointSpreadFunctionGaussian::get_size(float step_x, float step_y) const
{
    // FIXME

    return NDShape({util_num::roundu_odd(5*m_fwhm_x/step_x),
                    util_num::roundu_odd(5*m_fwhm_y/step_y)});
}

void PointSpreadFunctionGaussian::as_image(float step_x, float step_y, int size_x, int size_y, float* data) const
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

PointSpreadFunctionLorentzian::PointSpreadFunctionLorentzian(float fwhm)
    : PointSpreadFunctionLorentzian(fwhm, fwhm, 0)
{
}

PointSpreadFunctionLorentzian::PointSpreadFunctionLorentzian(float fwhm_x, float fwhm_y, float pa)
    : m_fwhm_x(fwhm_x)
    , m_fwhm_y(fwhm_y)
    , m_pa(pa)
{
}

PointSpreadFunctionLorentzian::~PointSpreadFunctionLorentzian()
{
}

NDShape PointSpreadFunctionLorentzian::get_size(float step_x, float step_y) const
{
    return NDShape({util_num::roundu_odd(5*m_fwhm_x/step_x),
                    util_num::roundu_odd(5*m_fwhm_y/step_y)});
}

void PointSpreadFunctionLorentzian::as_image(float step_x, float step_y, int size_x, int size_y, float* data) const
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

PointSpreadFunctionMoffat::PointSpreadFunctionMoffat(float fwhm, float beta)
    : PointSpreadFunctionMoffat(fwhm, fwhm, 0, beta)
{
}

PointSpreadFunctionMoffat::PointSpreadFunctionMoffat(float fwhm_x, float fwhm_y, float pa, float beta)
    : m_fwhm_x(fwhm_x)
    , m_fwhm_y(fwhm_y)
    , m_beta(beta)
    , m_pa(pa)
{
}

PointSpreadFunctionMoffat::~PointSpreadFunctionMoffat()
{
}

NDShape PointSpreadFunctionMoffat::get_size(float step_x, float step_y) const
{
    return NDShape({util_num::roundu_odd(5*m_fwhm_x/step_x),
                    util_num::roundu_odd(5*m_fwhm_y/step_y)});
}

void PointSpreadFunctionMoffat::as_image(float step_x, float step_y, int size_x, int size_y, float* data) const
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
