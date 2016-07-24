
#include "gbkfit/instrument.hpp"
#include "gbkfit/math.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/spread_functions.hpp"

namespace gbkfit
{

Instrument::Instrument(PointSpreadFunction* psf,
                       LineSpreadFunction* lsf)
    : m_psf(psf)
    , m_lsf(lsf)
{
}

Instrument::~Instrument()
{
}



NDShape Instrument::get_psf_size_spat(float step_x, float step_y) const
{
    return m_psf->get_size(step_x, step_y);
}

NDShape Instrument::get_psf_size_spec(float step_z) const
{
    return m_lsf->get_size(step_z);
}

NDShape Instrument::get_psf_size_cube(float step_x, float step_y, float step_z) const
{
    NDShape psf_shape = get_psf_size_spat(step_x, step_y);
    NDShape lsf_shape = get_psf_size_spec(step_z);
    return NDShape({psf_shape[0], psf_shape[1], lsf_shape[0]});
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_spat(float step_x, float step_y) const
{
    return m_psf->as_image(step_x, step_y);
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_spec(float step_z) const
{
    return m_lsf->as_array(step_z);
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_cube(float step_x, float step_y, float step_z) const
{
    std::unique_ptr<NDArrayHost> spat_data = create_psf_data_spat(step_x, step_y);
    std::unique_ptr<NDArrayHost> spec_data = create_psf_data_spec(step_z);
    std::unique_ptr<NDArrayHost> cube_data = create_psf_data_cube(spat_data, spec_data);
    return cube_data;
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_spat(float step_x, float step_y, int size_x, int size_y) const
{
    return m_psf->as_image(step_x, step_y, size_x, size_y);
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_spec(float step_z, int size_z) const
{
    return m_lsf->as_array(step_z, size_z);
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_cube(float step_x, float step_y, float step_z, int size_x, int size_y, int size_z) const
{
    std::unique_ptr<NDArrayHost> spat_data = create_psf_data_spat(step_x, step_y, size_x, size_y);
    std::unique_ptr<NDArrayHost> spec_data = create_psf_data_spec(step_z, size_z);
    std::unique_ptr<NDArrayHost> cube_data = create_psf_data_cube(spat_data, spec_data);
    return cube_data;
}

std::unique_ptr<NDArrayHost> Instrument::create_psf_data_cube(const std::unique_ptr<NDArrayHost>& data_spat,
                                                              const std::unique_ptr<NDArrayHost>& data_spec) const
{
    int size_x = data_spat->get_shape()[0];
    int size_y = data_spat->get_shape()[1];
    int size_z = data_spec->get_shape()[0];

    std::unique_ptr<NDArrayHost> cube_data = std::make_unique<NDArrayHost>(NDShape({size_x, size_y, size_z}));

    const float* spec_data_raw = data_spec->get_host_ptr();
    const float* spat_data_raw = data_spat->get_host_ptr();
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
