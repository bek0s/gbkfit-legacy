
#include "gbkfit/instrument.hpp"
#include "gbkfit/math.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/spread_functions.hpp"

namespace gbkfit
{

instrument::instrument(float step_x, float step_y, float step_z, point_spread_function* psf, line_spread_function* lsf)
    : m_step_x(step_x)
    , m_step_y(step_y)
    , m_step_z(step_z)
    , m_psf(psf)
    , m_lsf(lsf)
{
}

instrument::~instrument()
{
    delete m_psf;
    delete m_lsf;
}

float instrument::get_step_x(void) const
{
    return m_step_x;
}

float instrument::get_step_y(void) const
{
    return m_step_y;
}

float instrument::get_step_z(void) const
{
    return m_step_z;
}

NDShape instrument::get_recommended_psf_size_spec(void) const
{
    return get_recommended_psf_size_spec(m_step_z);
}

NDShape instrument::get_recommended_psf_size_spat(void) const
{
    return get_recommended_psf_size_spat(m_step_x, m_step_y);
}

NDShape instrument::get_recommended_psf_size_cube(void) const
{
    return get_recommended_psf_size_cube(m_step_x, m_step_y, m_step_z);
}

NDShape instrument::get_recommended_psf_size_spec(float step) const
{
    return m_lsf->get_recommended_size(step);
}

NDShape instrument::get_recommended_psf_size_spat(float step_x, float step_y) const
{
    return m_psf->get_recommended_size(step_x, step_y);
}

NDShape instrument::get_recommended_psf_size_cube(float step_x, float step_y, float step_z) const
{
    NDShape size_lsf = get_recommended_psf_size_spec(step_z);
    NDShape size_psf = get_recommended_psf_size_spat(step_x, step_y);

    return NDShape({size_psf[0], size_psf[1], size_lsf[0]});
}

NDArrayHost* instrument::create_psf_spec_data(void) const
{
    return m_lsf->as_array(m_step_z);
}

NDArrayHost* instrument::create_psf_spat_data(void) const
{
    return m_psf->as_image(m_step_x, m_step_y);
}

NDArrayHost* instrument::create_psf_cube_data(void) const
{
    NDArrayHost* spec_data = create_psf_spec_data();
    NDArrayHost* spat_data = create_psf_spat_data();
    NDArrayHost* cube_data = create_psf_cube_data(spec_data, spat_data);
    delete spec_data;
    delete spat_data;
    return cube_data;
}

NDArrayHost* instrument::create_psf_spec_data(int size) const
{
    return m_lsf->as_array(size, m_step_z);
}

NDArrayHost* instrument::create_psf_spat_data(int size_x, int size_y) const
{
    return m_psf->as_image(size_x, size_y, m_step_x, m_step_y);
}

NDArrayHost* instrument::create_psf_cube_data(int size_x, int size_y, int size_z) const
{
    NDArrayHost* spec_data = create_psf_spec_data(size_z);
    NDArrayHost* spat_data = create_psf_spat_data(size_x, size_y);
    NDArrayHost* cube_data = create_psf_cube_data(spec_data, spat_data);
    delete spec_data;
    delete spat_data;
    return cube_data;
}

NDArrayHost* instrument::create_psf_cube_data(float step_x, float step_y, float step_z) const
{
    NDArrayHost* spec_data = m_lsf->as_array(step_z);
    NDArrayHost* spat_data = m_psf->as_image(step_x, step_y);
    NDArrayHost* cube_data = create_psf_cube_data(spec_data, spat_data);
    delete spec_data;
    delete spat_data;
    return cube_data;
}

NDArrayHost* instrument::create_psf_spec_data(int size, float step) const
{
    return m_lsf->as_array(size, step);
}

NDArrayHost* instrument::create_psf_spat_data(int size_x, int size_y, float step_x, float step_y) const
{
    return m_psf->as_image(size_x, size_y, step_x, step_y);
}

NDArrayHost* instrument::create_psf_cube_data(int size_x, int size_y, int size_z, float step_x, float step_y, float step_z) const
{
    NDArrayHost* spec_data = create_psf_spec_data(size_z, step_z);
    NDArrayHost* spat_data = create_psf_spat_data(size_x, size_y, step_x, step_y);
    NDArrayHost* cube_data = create_psf_cube_data(spec_data, spat_data);
    delete spec_data;
    delete spat_data;
    return cube_data;
}

NDArrayHost* instrument::create_psf_cube_data(const NDArrayHost* spec_data, const NDArrayHost* spat_data) const
{
    int size_x = spat_data->get_shape()[0];
    int size_y = spat_data->get_shape()[1];
    int size_z = spec_data->get_shape()[0];

    NDArrayHost* cube_data = new NDArrayHost({size_x, size_y, size_z});

    const float* spec_data_raw = spec_data->get_host_ptr();
    const float* spat_data_raw = spat_data->get_host_ptr();
    float* cube_data_raw = cube_data->get_host_ptr();

    for(int z = 0; z < size_z; ++z)
    {
        for(int y = 0; y < size_y; ++y)
        {
            for(int x = 0; x < size_x; ++x)
            {
                int idx_spec = z;
                int idx_spat = y*size_x + x;
                int idx_cube = z*size_x*size_y + y*size_x + x;
                cube_data_raw[idx_cube] = spec_data_raw[idx_spec] * spat_data_raw[idx_spat];
            }
        }
    }

    math::normalize_integral(cube_data_raw, size_x*size_y*size_z);

    return cube_data;
}



} // namespace gbkfit
